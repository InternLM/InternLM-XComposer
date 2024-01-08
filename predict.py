# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import torch
from transformers import AutoModel, AutoTokenizer
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        torch.set_grad_enabled(False)
        self.model = (
            AutoModel.from_pretrained(
                "internlm/internlm-xcomposer-7b",
                cache_dir="model_cache",
                trust_remote_code=True,
            )
            .cuda()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "internlm/internlm-xcomposer-7b", trust_remote_code=True
        )
        self.model.tokenizer = tokenizer

    def predict(
        self,
        image: Path = Input(description="Input image.", default=None),
        text: str = Input(description="Input text."),
    ) -> str:
        """Run a single prediction on the model"""
        output = self.model.generate(text, str(image) if image else None)
        return output
