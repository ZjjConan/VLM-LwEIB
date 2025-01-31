GPU_ID=$1
SEED=$2

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# training
bash scripts/lweib/xd_train.sh ${SEED} x2d

# testing
for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenetv2 imagenet_sketch imagenet_a imagenet_r
do
    bash scripts/lweib/xd_test.sh ${DATASET} ${SEED} x2d 10
done