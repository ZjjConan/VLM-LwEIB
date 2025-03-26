GPU_ID=$1
SEED=$2

export CUDA_VISIBLE_DEVICES=${GPU_ID}
for DATASET in imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    bash scripts/lweib/base2new_train.sh ${DATASET} ${SEED} base2new
    bash scripts/lweib/base2new_test.sh ${DATASET} ${SEED} base2new 25
done
