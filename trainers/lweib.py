from collections import OrderedDict

import random
import os.path as osp
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from .gsl import gradient_scale_layer
from .gpt3_prompts import load_CuPL_templates

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, adapater_parser=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if adapater_parser == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, adapater_parser])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    

class DWConvNormAct(nn.Module):
    def __init__(self, d_model, k_size, dim):
        super().__init__()
        
        self.dim = dim
        if dim == 2:
            self.conv = nn.Conv2d(
                d_model, d_model, k_size, padding=k_size//2, 
                groups=d_model, bias=False
            )
        elif dim == 1:
            self.conv = nn.Conv1d(
                d_model, d_model, k_size, padding=k_size//2, 
                groups=d_model, bias=False
            )
        

    def forward(self, x):
        if self.dim == 1:
            x = x.permute(1,2,0)
            x = self.conv(x).permute(2,0,1)
            return x
        
        elif self.dim == 2:
            n_token, b_size, d_model = x.shape
            p_size = int(math.sqrt(n_token))
            x = x.permute(1,2,0)
            x = self.conv(x.reshape(b_size, d_model, p_size, p_size))
            x = x.reshape(b_size, d_model, p_size*p_size).permute(2,0,1)
            return x

class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # build multi-modal adapter
        self.text_adapter_parser = lambda x : self.return_text_adapter(x)
        self.text_adapted_func = lambda x, y, z: self.return_text_adapted_x(x, y, z)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.LWEIB.ADAPTER_START,
            cfg.TRAINER.LWEIB.ADAPTER_END,
            dtype=clip_model.dtype
        )
        
        self.visual_adapter_parser = lambda x : self.return_visual_adapter(x)
        self.visual_adapted_func = lambda x, y, z: self.return_visual_adapted_x(x, y, z)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.LWEIB.ADAPTER_START,
            cfg.TRAINER.LWEIB.ADAPTER_END,
            is_visual=True,
            dtype=clip_model.dtype
        )

        self._build_text_prompts(cfg, classnames)
        self.n_cls = len(classnames)
        self.adapter_scale = float(cfg.TRAINER.LWEIB.ADAPTER_SCALE)
        self.adapter_scale_factor = float(cfg.TRAINER.LWEIB.ADAPTER_SCALE_FACTOR)
        self.slow_fast_ratio = cfg.TRAINER.LWEIB.SLOW_FAST_RATIO

    def _build_text_prompts(self, cfg, classnames):
        text_ctx_init = cfg.TRAINER.LWEIB.TEXT_CTX_INIT
        # Prompt Contexts
        prompt_ctxs = load_CuPL_templates(cfg.DATASET.NAME)
        prompt_ctxs = {k.lower().replace("_", " "): v for k, v in prompt_ctxs.items()}
        classnames = [name.replace("_", " ") for name in classnames]
        tk_prompts = []
        for cname in classnames:
            suffix = prompt_ctxs[cname.lower().replace("_", " ")]
            prompts = [text_ctx_init + " " + cname + ", " + ctx for ctx in suffix]
            prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
            tk_prompts.append(prompts)

        self.register_buffer("tk_prompts", torch.stack(tk_prompts, dim=0))

    def _build_adapter(self, d_model, n_layers, l_start, l_end, is_visual=False, dtype=torch.float32):
        adapter = [None] * (n_layers + 1)
        channel = d_model * 4
        for i in range(l_start, l_end+1):
            if is_visual:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("att_conv", DWConvNormAct(d_model, 3, 2)),
                    ("mlp_conv", DWConvNormAct(channel, 3, 2))
                ]))
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("att_conv", DWConvNormAct(d_model, 3, 1)),
                    ("mlp_conv", DWConvNormAct(channel, 3, 1))
                ]))

        adapter = nn.ModuleList([a for a in adapter])
        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()

        return adapter


    def return_text_adapter(self, index):
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            adapter_scale = self.adapter_scale * self.adapter_scale_factor
        else:
            adapter_scale = self.adapter_scale
        return self.text_adapter[index], adapter_scale, self.text_adapted_func
    
    def return_text_adapted_x(self, x, adapter, scale):
        y = gradient_scale_layer(x, scale)
        y = adapter(y)
        y = gradient_scale_layer(y*scale, 1.0/scale)
        x = x + y
        return x

    def return_visual_adapter(self, index):
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            adapter_scale = self.adapter_scale * self.adapter_scale_factor
        else:
            adapter_scale = self.adapter_scale
        return self.visual_adapter[index], adapter_scale, self.visual_adapted_func
    
    def return_visual_adapted_x(self, x, adapter, scale):
        n_token = x.shape[0]
        # LNC -> NCL
        cls_token, x = torch.split(x, [1, n_token-1], dim=0)
        y = gradient_scale_layer(x, scale)
        y = adapter(y)
        y = gradient_scale_layer(y*scale, 1.0/scale)
        x = x + y
        x = torch.cat([cls_token, x], dim=0)
        return x

    def update_adapter_scale(self, scale_factor):
        self.adapter_scale = self.adapter_scale * scale_factor

    def forward(self):
        return self.text_adapter_parser, self.visual_adapter_parser

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

        self.all_cls = torch.arange(0, len(classnames))
        self.neg_sampling_ratio = cfg.TRAINER.LWEIB.NEG_SAMPLING_RATIO
        self.text_features_for_inference = None

    def encode_text(self, prompts, tk_prompts, adapter_parser=None):
        if adapter_parser is not None:
            text_features = self.text_encoder(
                prompts.type(self.dtype), tk_prompts.type(self.dtype), adapter_parser
            )
        else:
            text_features = self.text_encoder(
                prompts.type(self.dtype), tk_prompts.type(self.dtype)
            )
        return text_features
    
    def encode_image(self, image, adapter_parser=None):
        if adapter_parser is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), adapter_parser]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features


    def forward(self, image, label=None):
        text_adapter_parser, visual_adapter_parser = self.adapter_learner()

        if self.adapter_learner.training:

            tk_prompts = self.adapter_learner.tk_prompts
            n_cls, n_temp = tk_prompts.shape[0:2]

            if self.neg_sampling_ratio >= 0 and self.all_cls.shape[0] > image.size(0):
                # get positive prompts and samples' labels
                self.all_cls = self.all_cls.to(label.device)
                pos_c, inversed_c = torch.unique(label, return_inverse=True)
                pos_prompts = tk_prompts[pos_c]

                if self.neg_sampling_ratio > 0:
                    neg_c = [c not in pos_c for c in self.all_cls]
                    neg_prompts = tk_prompts[neg_c]
                    n_neg = min(neg_prompts.shape[0], len(pos_c) * self.neg_sampling_ratio)
                    i_neg = torch.multinomial(torch.ones(neg_prompts.shape[0]), n_neg)
                    neg_prompts = neg_prompts[i_neg]

                    tk_prompts = torch.cat([pos_prompts, neg_prompts], dim=0)
                else:
                    tk_prompts = pos_prompts
                    
                n_cls = tk_prompts.shape[0]
                label = inversed_c

            iid = torch.randint(0, n_temp, (1, n_cls), dtype=torch.long)
            prompts = tk_prompts[torch.arange(n_cls), iid].squeeze(0)

            with torch.no_grad():
                embedding = self.token_embedding(prompts)
                
            text_features = self.encode_text(embedding, prompts, text_adapter_parser)
            image_features = self.encode_image(image, visual_adapter_parser)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits, label
        else:
            if self.text_features_for_inference is not None:
                text_features = self.text_features_for_inference
            else:
                tk_prompts = self.adapter_learner.tk_prompts
                n_cls, n_temp = tk_prompts.shape[0:2]
                mean_text_features = 0
                for iid in range(n_temp):
                    prompts = tk_prompts[:, iid]
                    with torch.no_grad():
                        embedding = self.token_embedding(prompts)

                    text_features = self.encode_text(embedding, prompts, text_adapter_parser)
                    text_features = F.normalize(text_features, dim=-1)
                    mean_text_features += text_features
                mean_text_features /= n_temp
                self.text_features_for_inference = F.normalize(mean_text_features, dim=1)
                text_features = self.text_features_for_inference
            
            image_features = self.encode_image(image, visual_adapter_parser)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits


@TRAINER_REGISTRY.register()
class LwEIB(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LWEIB.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.LWEIB.PREC == "fp32" or cfg.TRAINER.LWEIB.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            if "text_adapter" not in name and "visual_adapter" not in name:
                param.requires_grad_(False)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.adapter_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LWEIB.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LWEIB.PREC
        if prec == "amp":
            with autocast():
                logits, label = self.model(image, label)
                loss = F.cross_entropy(logits, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, label = self.model(image, label)
            loss = F.cross_entropy(logits, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "n_cls": logits.shape[1]
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "text_features_for_inference" in state_dict:
                del state_dict["text_features_for_inference"]
            if "tk_prompts" in state_dict:
                del state_dict["tk_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)