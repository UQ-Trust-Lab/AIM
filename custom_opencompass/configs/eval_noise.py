from mmengine.config import read_base
from opencompass.models import HuggingFaceNoiseModel

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    #from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    #from opencompass.configs.datasets.demo.demo_math_base_gen import math_datasets
    #from opencompass.configs.datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets
    #from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    #from opencompass.configs.models.qwen.hf_qwen2_1_5b import models as hf_qwen2_1_5b_models
    #from opencompass.configs.models.hf_internlm.hf_internlm2_1_8b import models as hf_internlm2_1_8b_models

datasets = gsm8k_datasets + mmlu_datasets

models = []
for x in range(0, 32):
    std = 0.1 * x
    models.append(dict(
        type=HuggingFaceNoiseModel,
        abbr=f'qwen2-7b-hf-std-{std}',
        path='Qwen/Qwen-7B',
        max_out_len=2048,
        generation_kwargs= {"noise_std": std},
        batch_size=32,
        run_cfg=dict(num_gpus=1),
    ))
    models.append(dict(
        type=HuggingFaceNoiseModel,
        abbr=f'llama-2-7b-hf-std-{std}',
        path='meta-llama/Llama-2-7b-hf',
        max_out_len=2048,
        generation_kwargs= {"noise_std": std},
        batch_size=32,
        run_cfg=dict(num_gpus=1),
    ))

