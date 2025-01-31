#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys

sys.path.append(".")
import argparse
import contextlib
import copy
import functools
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from safetensors.torch import save_file, load_file,load_model

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    cast_training_params,
)
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid, convert_unet_state_dict_to_peft
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import json
import yaml
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file
from peft.utils import get_peft_model_state_dict, load_peft_weights
from data_utils.datasets import Sequence_Satellite_Dataset
from torch.utils.data import DataLoader, Dataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)
hf_token = ""

def log_validation(
    controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = SD3ControlNetModel.from_pretrained(
            train_config.output_dir, torch_dtype=weight_dtype
        )

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        train_config.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        revision=train_config.revision,
        variant=train_config.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(torch.device(accelerator.device))
    pipeline.set_progress_bar_config(disable=True)

    if train_config.seed is None:
        generator = None
    else:
        generator = torch.manual_seed(train_config.seed)

    if len(train_config.validation_image) == len(train_config.validation_prompt):
        validation_images = train_config.validation_image
        validation_prompts = train_config.validation_prompt
    elif len(train_config.validation_image) == 1:
        validation_images = train_config.validation_image * len(train_config.validation_prompt)
        validation_prompts = train_config.validation_prompt
    elif len(train_config.validation_prompt) == 1:
        validation_images = train_config.validation_image
        validation_prompts = train_config.validation_prompt * len(train_config.validation_image)
    else:
        raise ValueError(
            "number of `train_config.validation_image` and `train_config.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = (
        contextlib.nullcontext()
        if is_final_validation
        else torch.autocast(accelerator.device.type)
    )

    for validation_prompt, validation_image in zip(
        validation_prompts, validation_images
    ):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(train_config.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt,
                    control_image=validation_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "validation_image": validation_image,
                "images": images,
                "validation_prompt": validation_prompt,
            }
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                tracker.writer.add_image(
                    "Controlnet conditioning",
                    np.asarray([validation_image]),
                    step,
                    dataformats="NHWC",
                )

                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(
                    wandb.Image(validation_image, caption="Controlnet conditioning")
                )

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()

        if not is_final_validation:
            controlnet.to(accelerator.device)

        return image_logs


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=train_config.revision,
        variant=train_config.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=train_config.revision,
        variant=train_config.variant,
    )
    text_encoder_three = class_three.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=train_config.revision,
        variant=train_config.variant,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# SD3 controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
The weights were trained using [ControlNet](https://github.com/lllyasviel/ControlNet) with the [SD3 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sd3.md).
{img_str}

Please adhere to the licensing terms as described `[here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "controlnet",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompts_short: str,
    prompts_long: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompts_short = [prompts_short] if isinstance(prompts_short, str) else prompts_short
    prompts_long = [prompts_long] if isinstance(prompts_long, str) else prompts_long
    
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompts_short,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompts_long,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def main(train_config, caption_config):
    if train_config.report_to == "wandb" and train_config.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and train_config.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(train_config.output_dir, train_config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=train_config.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=train_config.mixed_precision,
        log_with=train_config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if train_config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if train_config.seed is not None:
        set_seed(train_config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if train_config.output_dir is not None:
            os.makedirs(train_config.output_dir, exist_ok=True)

        if train_config.push_to_hub:
            repo_id = create_repo(
                repo_id=train_config.hub_model_id or Path(train_config.output_dir).name,
                exist_ok=True,
                token=train_config.hub_token,
            ).repo_id
            
    logger.info("Train dataset loading started.")
            
    use_numerical_values = True
    masks = None
    is_train = True
    if caption_config != None:
        use_numerical_values = caption_config['use_numerical_values']
        masks = caption_config['masks']
        print("Values are initialized from caption config file.")

    dataset = Sequence_Satellite_Dataset(
        csv_paths = train_config.csv_paths, 
        dataset_names = train_config.dataset_names, 
        root = train_config.root_path, 
        past_month_max = 6,
        use_numerical_values = use_numerical_values, 
        masks = masks, 
        img_size = train_config.input_size, 
        is_train = is_train
    )
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size= train_config.per_gpu_batch_size, 
        num_workers = train_config.num_workers, 
        pin_memory = True
    )  
    
    logger.info("Train dataset loading complete.")
    
    logger.info("Loading CLIP Tokenizers 1")
    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=train_config.revision,
        token=hf_token
    )
    logger.info("CLIP Tokenizers 1 Loaded.")
    
    logger.info("Loading CLIP Tokenizers 2")
    tokenizer_two = CLIPTokenizer.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=train_config.revision,
        token=hf_token
        
    )
    logger.info("CLIP Tokenizers 2 Loaded.")
    
    logger.info("Loading T5 Tokenizers")
    tokenizer_three = T5TokenizerFast.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=train_config.revision,
        token=hf_token
        
    )
    logger.info("T5 Tokenizers Loaded.")

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        train_config.pretrained_model_name_or_path, train_config.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        train_config.pretrained_model_name_or_path, train_config.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        train_config.pretrained_model_name_or_path, train_config.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    logger.info("Loading Noise Scheduler")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        train_config.pretrained_model_name_or_path, subfolder="scheduler"
    )
    logger.info("Noise Scheduler Loaded.")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    logger.info("Loading Text Encoders")
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    logger.info("Text Encoders Loaded.")
    
    logger.info("Loading VAE")
    vae = AutoencoderKL.from_pretrained(
        train_config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=train_config.revision,
        torch_dtype=torch.float32
    )
    logger.info("VAE Loaded.")
    transformer_lora_config = LoraConfig(
            r=train_config.lora_rank,
            lora_alpha=train_config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    logger.info("Loading Transformer")

    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer", torch_dtype=torch.float16
    )
    transformer.add_adapter(transformer_lora_config)

    if not train_config.pretrained_single_image_transformer_path:
        print("ERROR provide pretrained_single_image_transformer_path")
        exit(0)
         #1024
    path = os.path.join(train_config.pretrained_single_image_transformer_path ,"transformer.safetensors")
    missing = load_model(transformer, path, device = "cuda")
           
                
    print("transformer.config:" , transformer.config)
    logger.info("Transformer Loaded.")

    logger.info("Initializing controlnet weights from transformer")
    controlnet = SD3ControlNetModel.from_transformer(
        transformer,
        num_layers=train_config.num_of_controlnet_layers,
        num_extra_conditioning_channels=train_config.num_extra_conditioning_channels,
    )

    print("controlnet.config:" , controlnet.config)
    logger.info("Controlnet initialized from transformer")

    controlnet.requires_grad_(False)
    
    controlnet.add_adapter(transformer_lora_config)

    if train_config.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        
        path = os.path.join(train_config.controlnet_model_name_or_path, "controlnet.safetensors")
        missing = load_model(controlnet, path, device = "cuda")

    else:
        logger.info("Loading controlnet weights from lora finetuned transformer")
        missing = load_model(controlnet, path, strict=False, device = "cuda")
    
    controlnet.requires_grad_(True)
    trainable_params = sum(p.numel() for p in controlnet.parameters() )
    print("controlnet trainable_params after setting requeires grad to true", trainable_params) #1 billion + 18 
    
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    controlnet.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                controlnet_lora_layers_to_save = None
                for model in models:
                    if isinstance(model, type(unwrap_model(controlnet))):
                        all_layers = model.state_dict()
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
                save_file(all_layers, os.path.join(output_dir,"controlnet.safetensors"))

                

        def load_model_hook(models, input_dir):
            controlnet_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(controlnet))):
                    controlnet_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                path = os.path.join(input_dir, "controlnet.safetensors")
                missing = load_model(controlnet_, path, device = 'cuda')
                with open('missing.txt', 'w') as file:
                    file.write(str(missing))
            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804

            controlnet_.to(accelerator.device, dtype=weight_dtype)


            if  accelerator.mixed_precision == "fp16":
                models = [controlnet_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if train_config.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if train_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if train_config.scale_lr:
        train_config.learning_rate = (
            train_config.learning_rate
            * train_config.gradient_accumulation_steps
            * train_config.per_gpu_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if train_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation

    controlnet.controlnet_blocks.requires_grad_(True)
    controlnet.pos_embed_input.requires_grad_(True)
    params_to_optimize = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    print("******************************************************")
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f" number of trainable params after controlnet blocks : {trainable_params}")
    if accelerator.mixed_precision == "fp16":
        models = [controlnet]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    logger.info("Models moved to device and cast to weight_dtype.")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        weight_decay=train_config.adam_weight_decay,
        eps=train_config.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    if train_config.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
        print("we upcast vae ***********************************************")
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_config.gradient_accumulation_steps
    )
    if train_config.max_train_steps is None:
        train_config.max_train_steps = train_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        train_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=train_config.max_train_steps * accelerator.num_processes,
        num_cycles=train_config.lr_num_cycles,
        power=train_config.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        train_config.max_train_steps = train_config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_config.num_train_epochs = math.ceil(train_config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     tracker_config = dict(vars(args))

    #     # tensorboard cannot handle list types for config
    #     tracker_config.pop("validation_prompt")
    #     tracker_config.pop("validation_image")

    #     accelerator.init_trackers(train_config.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        train_config.per_gpu_batch_size
        * accelerator.num_processes
        * train_config.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {train_config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_config.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {train_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if train_config.resume_from_checkpoint:
        if train_config.resume_from_checkpoint != "latest":
            path = os.path.basename(train_config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(train_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{train_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(train_config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            updated_weights_2= {name: p.clone().detach() for name, p in controlnet.named_parameters() if p.requires_grad}
            with open('weights_2.txt', 'w') as file:
                for name, weight in updated_weights_2.items():
                    file.write(f'{name}: {weight}\n')


    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, train_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None
    for epoch in range(first_epoch, train_config.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                pixel_values = batch["img"].to(dtype=vae.dtype).to(accelerator.device, non_blocking=True)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (
                    model_input - vae.config.shift_factor
                ) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=train_config.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=train_config.logit_mean,
                    logit_std=train_config.logit_std,
                    mode_scale=train_config.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                prompts_short = batch["caption_clip"]
                prompts_long = batch["caption_t5"]
                prompts_short_control = batch["control_caption_clip"]
                prompts_long_control = batch["control_caption_t5"]
                max_sequence_length = 256

                (
                    prompt_embeds,
                    pooled_prompt_embeds
                ) = encode_prompt(
                    text_encoders,
                    tokenizers,
                    prompts_short,
                    prompts_long,
                    max_sequence_length=train_config.max_sequence_length,
                )
                
                (
                    control_prompt_embeds,
                    control_pooled_prompt_embeds
                ) = encode_prompt(
                    text_encoders,
                    tokenizers,
                    prompts_short_control,
                    prompts_long_control,
                    max_sequence_length=train_config.max_sequence_length,
                )
                # controlnet(s) inference
                controlnet_image = batch["past_img"].to(
                    dtype=vae.dtype
                )
                controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
                controlnet_image = controlnet_image * vae.config.scaling_factor

                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=control_prompt_embeds,
                    pooled_projections=control_pooled_prompt_embeds,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )[0]
                control_block_res_samples = [
                    sample.to(dtype=weight_dtype)
                    for sample in control_block_res_samples
                ]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if train_config.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=train_config.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                if train_config.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, train_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % train_config.checkpointing_steps == 0 or global_step == train_config.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if train_config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(train_config.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= train_config.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - train_config.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        train_config.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            train_config.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        train_config.validation_prompt is not None
                        and global_step % train_config.validation_steps == 0
                    ):
                        image_logs = log_validation(
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= train_config.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet = controlnet.to(torch.float32)
        all_layers = controlnet.state_dict()
        save_file(all_layers, os.path.join(train_config.output_dir,"controlnet.safetensors"))

        # Run a final round of validation.
        image_logs = None
        if train_config.validation_prompt is not None:
            image_logs = log_validation(
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if train_config.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=train_config.pretrained_model_name_or_path,
                repo_folder=train_config.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=train_config.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
    
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            # Recursively convert dictionaries to Config objects
            if isinstance(value, dict):
                value = Config(value)
            # Preserve float for scientific notation
            elif isinstance(value, str):
                try:
                    # Convert string to float if it represents a valid float
                    value = float(value) if 'e' in value or '.' in value else value
                except ValueError:
                    pass
            setattr(self, key, value)
            # print(f"Config attribute set: {key} (type: {type(value)})")

    def __getattr__(self, name):
        # Return None if an attribute does not exist
        return None

def copy_file(src, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # Create the output folder if it doesn't exist
    shutil.copy(src, dest_folder)  

def read_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path, "r") as file:
            config = json.load(file)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            config["output_dir"] = os.path.join(config["output_dir"], config["tracker_project_name"])
            print(config["output_dir"])

            # Convert YAML data to Config object
            config = Config(config)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model using a config file")
    parser.add_argument(
        "--config_train",
        type=str,
        required=True,
        help="Path to the config file (json or yaml)",
    )
    parser.add_argument(
        "--config_caption",
        type=str,
        default=None,
        help="Path to the config file (json or yaml)",
    )
    args = parser.parse_args()

    train_config = read_config(args.config_train)
    caption_config = None
    if args.config_caption != None:
       caption_config = read_config(args.config_caption)
    print(caption_config)
    main(train_config, caption_config)
    
    copy_file(args.config_train, args.output_dir)
    if args.config_caption != None: 
        copy_file(args.config_caption, output_dir)
