from caltech101 import CALTECH101_TEMPLATES
from oxford_pets import OXFORD_PETS_TEMPLATES
from stanford_cars import STANFORD_CARS_TEMPLATES
from oxford_flowers import OXFORD_FLOWERS_TEMPLATES
from food101 import FOOD101_TEMPLATES
from fgvc_aircraft import FGVC_AIRCRAFT_TEMPLATES
from sun397 import SUN397_TEMPLATES
from dtd import DTD_TEMPLATES
from eurosat import EUROSAT_TEMPLATES
from ucf101 import UCF101_TEMPLATES
from imagenet import IMAGENET_TEMPLATES

def load_CuPL_templates(dataset_name):
    dname = dataset_name.lower()
    if dname == "caltech101":
        return CALTECH101_TEMPLATES
    elif dname == "oxfordpets":
        return OXFORD_PETS_TEMPLATES
    elif dname == "stanfordcars":
        return STANFORD_CARS_TEMPLATES
    elif dname == "oxfordflowers":
        return OXFORD_FLOWERS_TEMPLATES
    elif dname == "food101":
        return FOOD101_TEMPLATES
    elif dname == "fgvcaircraft":
        return FGVC_AIRCRAFT_TEMPLATES
    elif dname == "describabletextures":
        return DTD_TEMPLATES
    elif dname == "eurosat":
        return EUROSAT_TEMPLATES
    elif dname == "sun397":
        return SUN397_TEMPLATES
    elif dname == "ucf101":
        return UCF101_TEMPLATES
    elif "imagenet" in dname:
        return IMAGENET_TEMPLATES

import json

for dataset_name in ("caltech101", "oxfordpets", "stanfordcars", "oxfordflowers", \
                     "food101", "fgvcaircraft", "describabletextures", "eurosat", \
                     "sun397", "ucf101", "imagenet"):
    prompt_ctxs = load_CuPL_templates(dataset_name)
    for k in prompt_ctxs:
        prompt_ctxs[k] = prompt_ctxs[k][:20]
        print(k, len(prompt_ctxs[k]))

    with open(dataset_name + '_after.py', 'w', encoding='utf-8') as f:
        json.dump(prompt_ctxs, f, ensure_ascii=False, indent=4)