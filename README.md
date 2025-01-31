# EcoMapper: Generative Modeling for Climate-Aware Satellite Imagery

## Introduction
EcoMapper is a powerful framework for generative modeling, designed to create climate-aware satellite imagery. Using the latest advancements in diffusion models, it leverages the power of fine-tuned Stable Diffusion 3 (SD3) with LoRA (Low-Rank Adaptation) and ControlNet to generate high-quality, contextually aware satellite images based on diverse environmental prompts.

## Installation
To get started, set up the environment and install the required dependencies by following the steps below:


First, clone the repository to your local machine and then create an environment to install requirements.

```bash
git clone https://github.com/anonymous-submission-xxx/ecomapper.git
cd Ecomapper
conda create -n ecomapper python=3.10
conda activate ecomapper
pip install -r requirements.txt
```

Note: For diffusionsat related activity, visit https://github.com/samar-khanna/DiffusionSat and follow the corresponding instructions to create an appropriate environment.

## Training
EcoMapper provides different versions for training models with or without LoRA, as well as for single image generation and ControlNet. For each training session, two configuration files are needed: one for model architecture, dataset and training parameters (`config_train`), and another for the masks used for prompting (`config_caption`).

### Single Image Generation
You can train the model with or without LoRA by selecting the appropriate configuration files.

#### SD3 without LoRA
To train the model without LoRA, use the following command:
```bash
accelerate launch --mixed_precision=fp16 train.py --config_train configs/train/train.yaml --config_caption configs/caption/skip_date_location.json
```
#### SD3 with LoRA
To train the model with LoRA, use the following command:
```bash
accelerate launch --mixed_precision=fp16 train_lora.py --config_train configs/train/train_lora.yaml --config_caption configs/caption/skip_date_location.json
```

### ControlNet Image Generation
For ControlNet image generation, the process is similar: you can train the model with or without LoRA by selecting the corresponding configuration files.


#### ControlNet SD3 without LoRA
To train ControlNet without LoRA, use this command:
```bash
accelerate launch --mixed_precision=fp16 train_controlnet.py --config_train configs/train/train_controlnet.yaml --config_caption configs/caption/skip_date_location.json
```
#### ControlNet SD3 with LoRA
To train ControlNet with LoRA, run this command:
```bash
accelerate launch --mixed_precision=fp16 train_controlnet_lora.py --config_train configs/train/train_controlnet_lora.yaml --config_caption configs/caption/skip_date_location.json
```

### Inference
After training, use the validation script to generate results from the trained model. Ensure that you specify the correct model paths in the script.
#### Inference for ControlNet SD3 without LoRA

```bash
python validation/validate_controlnet.py
```
#### Inference for ControlNet SD3 with LoRA

```bash
python validation/validate_controlnet_lora.py
```

#### Computing the metrics (after validation scripts)

Notes for diffsat metrics: To compute clip score, before running batch_compute_metrics.py, insert the clip caption from the json files of the SD3 images to the json file of the diffsat images.

```bash
python validation/batch_compute_metrics.py
```
