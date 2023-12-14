# Model Zoo

If you are interested in including any other details in Model Zoo, please open an issue :)

The usage of ShareGPT4V checkpoints should comply with the base LLM's model license: [Llama 2](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).

## ShareGPT4V models

| Name | LLM | Checkpoint | LLaVA-Bench-Wild | MME-perception | MME-cognition | MMBench | MMBench-CN | SEED-image | MM-Vet | QBench | SQA-image | VQA-v2 | VizWiz | GQA | TextVQA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ShareGPT4V-7B | Vicuna-7B | [ShareGPT4V-7B](https://huggingface.co/Lin-Chen/ShareGPT4V-7B) | 72.6 | 1567.4 | 376.4 | 68.8 | 62.2 | 69.7 | 37.6 | 63.4 | 68.4 | 80.6 | 57.2 | 63.3 | 60.4 |
| ShareGPT4V-13B | Vicuna-13B | [ShareGPT4V-13B](https://huggingface.co/Lin-Chen/ShareGPT4V-13B) | 79.9 | 1618.7 | 303.2 | 68.5 | 63.7 | 70.8 | 43.1 | 65.2 | 71.2 | 81.0 | 55.6 | 64.8 | 62.2 |

## Pretrained Vision Encoders

These are vision encoder weights we have pretrained. You can use these weights for our or your own visual instruction tuning. They are just pretrained on ShareGPT4V-PT image-text pairs and are NOT instruction-tuned, which means they do NOT follow instructions as well as our official models and can output repetitive, lengthy, and garbled outputs. If you want to have nice conversations with ShareGPT4V models, use the checkpoints above (in ShareGPT4V models).

| Base LLM | Vision Encoder | Projection | Pretrain Data | Pretraining schedule | Download |
|----------|----------------|---------------|----------------------|----------|----------|
| Vicuna-13B-v1.5 | CLIP-L-336px-ft-l12 | MLP-2x | ShareGPT4V-PT-1.2M | 1e | [vision encoder](https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12) |
| Vicuna-7B-v1.5 | CLIP-L-336px-ft-l12 | MLP-2x | ShareGPT4V-PT-1.2M | 1e | [vision encoder](https://huggingface.co/Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12) |

## Pretrained Projector and LLM

These are projector and LLM weights we have pretrained. You can use these weights for our or your own visual instruction tuning. They are just pretrained on ShareGPT4V-PT image-text pairs and are NOT instruction-tuned, which means they do NOT follow instructions as well as our official models and can output repetitive, lengthy, and garbled outputs. If you want to have nice conversations with ShareGPT4V models, use the checkpoints above (in ShareGPT4V models).

| Base LLM | Vision Encoder | Projection | Pretrain Data | Pretraining schedule | Download |
|----------|----------------|---------------|----------------------|----------|----------|
| Vicuna-13B-v1.5 | CLIP-L-336px-ft-l12 | MLP-2x | ShareGPT4V-PT-1.2M | 1e | [projector and LLM](https://huggingface.co/Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5) |
| Vicuna-7B-v1.5 | CLIP-L-336px-ft-l12 | MLP-2x | ShareGPT4V-PT-1.2M | 1e | [projector and LLM](https://huggingface.co/Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12_vicuna-7b-v1.5) |
