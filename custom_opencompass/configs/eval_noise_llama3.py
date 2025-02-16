from mmengine.config import read_base
from opencompass.models import HuggingFaceNoiseModel

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets

datasets = gsm8k_datasets + mmlu_datasets

models = []
for x in range(0, 21):
    std = x * 0.1
    models.append(dict(
        type=HuggingFaceNoiseModel,
        abbr=f'llama-31-8b-Instruct-std-{std}',
        path='meta-llama/Llama-3.1-8B-Instruct',
        model_kwargs = {"revision": "0e9e39f"},
        max_out_len=1024,
        generation_kwargs= {'noise_std': std},
        batch_size=4,
        run_cfg=dict(num_gpus=1),
    ))
