gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

echo "Running on GPU list: ${gpu_list}"

ln -s ../internlm-xcomposer2d5-ol-7b internlm-xcomposer2d5-ol-7b

CUDA_VISIBLE_DEVICES=${GPULIST[0]} python backend_vs.py &
CUDA_VISIBLE_DEVICES=${GPULIST[0]} python backend_llm.py &
CUDA_VISIBLE_DEVICES=${GPULIST[0]} python backend.py
