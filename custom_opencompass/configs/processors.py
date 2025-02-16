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
