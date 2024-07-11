export HF_MODEL=internlm/internlm-xcomposer2d5-7b-4bit
export WORK_DIR=./internlm-xcomposer2d5-7b-4bit

lmdeploy lite auto_awq \
   $HF_MODEL \
  --work-dir $WORK_DIR