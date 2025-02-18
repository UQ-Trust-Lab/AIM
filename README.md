# AIM: AI Model Modulation with Logits Redistribution

This repository contains the implementations of our paper:  **"AI Model Modulation with Logits Redistribution"**.

## Introduction

Large-scale models are typically adapted to meet the diverse requirements of model owners and users. However, maintaining multiple specialized versions of the model is inefficient. In response, we propose AIM, a novel model modulation paradigm that enables a single model to exhibit diverse behaviors to meet the specific end requirements. AIM enables two key modulation modes: utility and focus modulations. The former provides model owners with dynamic control over output quality to deliver varying utility levels, and the latter offers users precise control to shift model’s focused input features. AIM introduces a logits redistribution strategy that operates in a training data-agnostic and retraining-free manner. We establish a formal foundation to ensure AIM’s regulation capability, based on the statistical properties of logits ordering via joint probability distributions. Our evaluation confirms AIM’s practicality and versatility for AI model modulation, with tasks spanning image classification, semantic segmentation and text generation, and prevalent architectures including ResNet, SegFormer and Llama.

## CV Task1: Image Classification
### 1. Project Structure

```
img_classification
├── dataset/
└── model/
└── val.py
            
```

### 2. Quick Start
* Install dependencies
```python
pip install torch torchvision matplotlib 
```
* Run evaluation (CIFAR-10/100 example)
For CIFAR10/CIFAR100, we use pretrained models from: ***[PyTorch CIFAR Models](https://github.com/chenyaofo/pytorch-cifar-models).*** 
```python
python img_classification/val.py --dataset cifar10 --model resnet56 --batch_size 128 --num_perturb_steps 101 --perturb_step 0.2 --plot --save_plot
```
- Results from Our paper for reference:
<br><img width="380" alt="image" src="https://github.com/user-attachments/assets/77c6881f-23b6-4679-8ff1-50f628e12329" />



## CV Task2: Semantic Segmentation
### 1. Project Structure

```
semantic-segmentation-main/
├── assests/                 # Test images
├── configs/                # Configuration files
├── data/                   # Dataset root
│   └── (example) ADEChallengeData2016/
│       ├── annotations/
│       └── images/
├── tools/perturbation/
│          ├── models.py                      # Perturbation models
│          ├── evaluation.py                  # Metric evaluation
│          └── infer_perturbation.py          # Inference script
└── output/                 # Results directory
            
```

### 2. Setup Instructions
- Please download base from ***[Semantic Segmentation](https://github.com/sithu31296/semantic-segmentation/tree/main)*** or Clone base repository
```bash
git clone https://github.com/sithu31296/semantic-segmentation.git
```
- Set up env
```python
pip install -e .
```
- Navigate to the base directory
```bash
cd semantic-segmentation
```
```bash
git clone https://github.com/CurtisYoung18/AIM.git
```
- ***Copy and Paste*** our segmentaion py files to semantic-segmentation-main/tools/
```bash
cp -r AIM-main/semantic_segmentation/perturbation tools/
```
- Download datasets(*e.g.,* ***[ADE20K datasets](http://sceneparsing.csail.mit.edu/)***). Place the dataset in the root directory and update the dataset path in configs accordingly
- Download the backbone model(eg. ***[Backbones](https://github.com/sithu31296/semantic-segmentation/blob/main/docs/BACKBONES.md), [Pretrained Models](https://github.com/sithu31296/semantic-segmentation/blob/main/docs/MODELS.md)***). Update the configuration file accordingly
- Add configs to desired yaml files:
```python
PERTURBATION:
  METHOD: 'single'           # Options: 'single' or 'multi'
  RATIO: 0.2                # Base perturbation ratio
  STEP_SIZE: 0.4            # Ratio increment step
  STEP_LIMIT: 10            # Number of perturbation steps
  TARGET_IDX: 136           # Class index for single perturbation
  POSITIVE: true            # Perturbation direction
  TARGETS:                  # For multi-class perturbation
    - index: 136
      positive: true
    - index: 20
      positive: false
```
- (Optional)Download inference pictures(*e.g.,* ***[Kitty](https://www.kaggle.com/datasets/klemenko/kitti-dataset)***) for testing and place them to assests(eg. /ade)


### 3.Run
<details open>
  <summary><strong>Inference</strong></summary>

To make an inference, edit the parameters of the config file from below.
* Change `MODEL` >> `NAME` and `BACKBONE` to your desired pretrained model.
* Change `DATASET` >> `NAME` to the dataset name depending on the pretrained model.
* Set `TEST` >> `MODEL_PATH` to pretrained weights of the testing model.
* Change `TEST` >> `FILE` to the file or image folder path you want to test.
* Testing results will be saved in `SAVE_DIR`.

- Example:
```python
python tools/perturbation/infer_perturbation.py \
  --cfg configs/ade20k.yaml \
  --pert_ratio 0.2 \
  --pert_method single
```
</details>

<details open>
<summary><strong>Evaluation</strong></summary>

```python
tools/perturbation/evaluation.py --cfg configs/ade20k.yaml
```
</details>

- Results from Our paper for reference:
<br> ![image](https://github.com/user-attachments/assets/115bab21-27c3-4bdc-85dd-1e12a5fd1f50)


### 4. Core Components

| File                  | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| `models.py`          | Implements SingleClassPerturbator and MultiClassPerturbator   |
| `evaluation.py`      | Evaluation is performed over a range of perturbation ratios on the "person" class.        |
| `infer_perturbation.py` | Visualization pipeline with perturbation controls         |





# NLP Task: Model Modulation during Text Generation
We tailored the OpenCompass framework to include a new class `HuggingFaceNoiseModel` which introduces the capability to inject Gaussian noise into the logits during the generation process of a Hugging Face model, thus enable model modulation for AI Models.  

## Key Modifications

Gaussian Noise Injection:

The `GaussianNoiseLogitsProcessor` class is responsible for adding Gaussian noise to the logits (scores) during the generation process. The noise is sampled from a normal distribution with a configurable mean and standard deviation (std).

Integration with Generation Configuration:

The HuggingFaceNoiseModel class checks for the presence of a noise_std parameter in the generation_kwargs. If found, it initializes the GaussianNoiseLogitsProcessor with the specified standard deviation and adds it to the LogitsProcessorList.

The noise_std parameter is then removed from generation_kwargs to avoid conflicts during the generation process.

OpencCompass Config:

THe main script is eval_noise_llama3.py, which contains the model, datasets, and std config. You can reproduce our experiments following Usage step by step.


## Usage
To use the HuggingFaceNoiseModel class, follow these steps:

### 🛠️ Installation

Below are the steps for quick installation and datasets preparation.

#### 💻 Environment Setup

We highly recommend using conda to manage your python environment.

- #### Create your virtual environment

  ```bash
  conda create --name opencompass python=3.10 -y
  conda activate opencompass
  ```

- #### Install OpenCompass via pip

  ```bash
    pip install -U opencompass

    ## Full installation (with support for more datasets)
    # pip install "opencompass[full]"

    ## Environment with model acceleration frameworks
    ## Manage different acceleration frameworks using virtual environments
    ## since they usually have dependency conflicts with each other.
    # pip install "opencompass[lmdeploy]"
    # pip install "opencompass[vllm]"

    ## API evaluation (i.e. Openai, Qwen)
    # pip install "opencompass[api]"
  ```

- #### Install OpenCompass from source

  If you want to use opencompass's latest features, or develop new features, you can also build it from source

  ```bash
    git clone https://github.com/open-compass/opencompass opencompass
    cd opencompass
    pip install -e .
    # pip install -e ".[full]"
    # pip install -e ".[vllm]"
  ```
- ### Run command
  ```bash
    python -u run.py configs/eval_noise_llama3.py
  ```
- #### Config file `eval_noise.py`
```python
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
```
