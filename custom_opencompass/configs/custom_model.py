from typing import List
from opencompass.models import HuggingFaceBaseModel
from transformers import LogitsProcessor, LogitsProcessorList
import torch

class GaussianNoiseLogitsProcessor(LogitsProcessor):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, input_ids, scores):
        # 添加高斯噪声到 logits (即 scores)
        noise = torch.normal(mean=self.mean, std=self.std, size=scores.shape).to(scores.device)
        return scores + noise

class HuggingFaceNoiseModel(HuggingFaceBaseModel):

    def __init__(self, path: str, model_kwargs: dict = ..., tokenizer_path: str | None = None, tokenizer_kwargs: dict = ..., peft_path: str | None = None, peft_kwargs: dict = ..., tokenizer_only: bool = False, generation_kwargs: dict = ..., max_seq_len: int | None = None, pad_token_id: int | None = None, stop_words: str | None = ..., **other_kwargs):
        self.model_kwargs = model_kwargs
        if "noise_std" in self.generation_kwargs:
            logits_processor_list = LogitsProcessorList()
            logits_processor_list.append(GaussianNoiseLogitsProcessor(std=self.generation_kwargs["noise_std"]))
            generation_kwargs["logits_processor"] = logits_processor_list
        super().__init__(path, model_kwargs, tokenizer_path, tokenizer_kwargs, peft_path, peft_kwargs, tokenizer_only, generation_kwargs, max_seq_len, pad_token_id, stop_words, **other_kwargs)
