export TOKENIZERS_PARALLELISM=False

# ds=librispeech
# ds=wenet_test_meeting
ds=wenet_test_net
checkpoint='internlm-xcomposer2d5-ol-7b/audio'
NPROC_PER_NODE=8

python -m torch.distributed.launch --use_env \
    --nproc_per_node ${NPROC_PER_NODE:-8} --nnodes 1 \
    evaluate_asr.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 20 \
    --num-workers 4