import sys

sys.path.append("..")
import warnings

# Suppress all warnings
warnings.simplefilter("ignore")

from accelerate import DistributedDataParallelKwargs

import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig, CLIPModel
import copy
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    SD3ControlNetModel,
    StableDiffusion3ControlNetPipeline,
)
from torch.utils.data import DataLoader, Dataset

from peft import LoraConfig, set_peft_model_state_dict

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
import bitsandbytes as bnb
from tqdm import tqdm
import argparse
import logging
import os
import random
import json
import yaml
import torch
import transformers
import diffusers
import shutil
import math
import numpy as np
from copy import deepcopy
from diffusers.optimization import get_scheduler
from shapely.geometry import shape as shapey
from shapely.wkt import loads as shape_loads
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
import random
from diffusers.utils.torch_utils import is_compiled_module

from data_utils.datasets import Satellite_Dataset

from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.tensorboard import SummaryWriter

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
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")
    return config


def freeze_model_params(model, config):

    if config["MM_DiT_blocks"] == "full":
       freeze_layer = 0
    if config["MM_DiT_blocks"] == "large":
       freeze_layer = 8
    if config["MM_DiT_blocks"] == "medium":
       freeze_layer = 12
    if config["MM_DiT_blocks"] == "small":
       freeze_layer = 17
    i = 0
    for name, layer in model.named_children():
        if "transformer_blocks" in name:

            for sublayer in layer.children():
                i += 1
                if i > freeze_layer:
                    sublayer.requires_grad_(True)
                else:
                    sublayer.requires_grad_(False)
        else:
            layer.requires_grad_(False)
    if config['time_text_embedding_layer']:
        model.time_text_embed.requires_grad_(True)
    if config['context_embedder_layer']:
        model.context_embedder.requires_grad_(True)
    if config['norm_out_layer']:
        model.norm_out.requires_grad_(True)
    if config['projection_layer']:
        model.proj_out.requires_grad_(True)

    params_list = [p for p in model.parameters() if p.requires_grad]
    return params_list


def train(config, config_caption):
    controlnet = None
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    output_dir = os.path.join(config["output_dir"], config["model_name"])

    load_path = id
    vae = AutoencoderKL.from_pretrained(
        load_path, subfolder="vae", torch_dtype=torch.float16
    )
    if config["load_path"] != "":
        load_path = config["load_path"]
    print("load_path", load_path)
  
    transformer = SD3Transformer2DModel.from_pretrained(
        load_path, subfolder="transformer", torch_dtype=torch.float16
    )
    print(transformer.config)
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        id, subfolder="scheduler"
    )
    last_iteration = 0
    checkpoint_info_path = os.path.join(output_dir, "log.json")
    if os.path.exists(checkpoint_info_path):
        with open(checkpoint_info_path, "r") as f:
            data = json.load(f)
            last_iteration = data["iteration"]


    accelerator_project_config = ProjectConfiguration(
        project_dir=config["project_dir"],
        logging_dir=os.path.join(
            config["logging_dir"], config["model_name"] + config["tracker_name"]
        ),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="fp16",
        log_with=config["log_with"],
        project_config=accelerator_project_config,
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(config["seed"])

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusion3Pipeline.from_pretrained(
            id, transformer=None, vae=None, torch_dtype=weight_dtype
    )

    pipe.text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_3.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_3.eval()
    pipe.text_encoder_2.eval()
    pipe.text_encoder.eval()
    
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()

    #######################################################
    if config['train_all_layers']:
        transformer.requires_grad_(True)
        parameters_list = [p for p in transformer.parameters() if p.requires_grad]
    else:
        transformer.requires_grad_(False)
        parameters_list = freeze_model_params(transformer, config)

    
    if accelerator.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    if config['use_8bit_adam']:
        optimizer_cls = bnb.optim.AdamW8bit  # torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    learning_rate = float(config["learning_rate"])
    optimizer = optimizer_cls(
        parameters_list,
        lr=learning_rate,
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_epsilon"]),
    )


    dataset_names = config["dataset_names"]
    root_path = config['dataset_root']
    csv_paths = config['csv_paths']
    dataset_names = config['dataset_names']
    input_size = config['input_size']
    per_gpu_batch_size = config['per_gpu_batch_size']
    num_workers = config['num_workers']
    use_numerical_values = True
    masks = None
    is_train = True
    if config_caption != None:
        use_numerical_values = config_caption['use_numerical_values']
        masks = config_caption['masks']

    dataset = Satellite_Dataset(csv_paths, dataset_names, root_path, use_numerical_values, masks, input_size, is_train)
    train_loader = DataLoader(dataset, batch_size= per_gpu_batch_size, num_workers = num_workers, pin_memory = True)  

    max_epochs = config["max_epochs"]

    max_steps = max_epochs * len(train_loader) // accelerator.num_processes
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps * accelerator.num_processes,
    ) 
    transformer, train_loader, optimizer, lr_scheduler = accelerator.prepare(
        transformer, train_loader, optimizer, lr_scheduler
    )

    device = torch.device("cuda")
    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(
        range(global_step, max_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    weighting_scheme = config["weighting_scheme"]
    output_dir = os.path.join(config["output_dir"], config["model_name"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / config["gradient_accumulation_steps"]
    )
    if config["resume_from_checkpoint"] != "":
        if config["resume_from_checkpoint"] != "latest":
            path = os.path.basename(config["resume_from_checkpoint"])
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config['resume_from_checkpoint']}' does not exist. Starting a new training run."
            )
            config["resume_from_checkpoint"] = ""
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * config["gradient_accumulation_steps"]
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * config["gradient_accumulation_steps"]
            )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(config["log_with"], tracker_config)

    #########################################################
    total_batch_size = (
        config["per_gpu_batch_size"]
        * accelerator.num_processes
        * config["gradient_accumulation_steps"]
    )

    logger.info("***** Running training *****")
    if controlnet is None:
        logger.info(
            f"  Trainable params  = {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
        )
    else:
        logger.info(
            f"  Trainable params  = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) + sum(p.numel() for p in controlnet.parameters())}"
        )
    logger.info(f"  Num Epochs = {config['max_epochs']}")
    logger.info(
        f"  Instantaneous batch size per device = {config['per_gpu_batch_size']}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}"
    )
    logger.info(f"  Total optimization steps = {max_steps}")

    #########################################################
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float16):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # print("3", controlnet.dtype)

    for epoch in range(first_epoch, max_epochs + first_epoch):

        train_loss = 0.0

        transformer.train()

        for step, batch in enumerate(train_loader):
            if (
                config["resume_from_checkpoint"] != ""
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % config["gradient_accumulation_steps"] == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(transformer):

                images = (
                    batch["img"]
                    .to(weight_dtype)
                    .to(accelerator.device, non_blocking=True)
                )          
                prompts_short = batch["caption_clip"]
                prompts_long = batch["caption_t5"]

                max_sequence_length = 256
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt=prompts_short,
                    prompt_2=prompts_short,
                    prompt_3=prompts_long,
                    max_sequence_length=max_sequence_length,
                )
             
                prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)

                model_input = vae.encode(images).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=weighting_scheme,
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (
                    u * noise_scheduler.config.num_train_timesteps
                ).long()
                timesteps = noise_scheduler.timesteps[indices].to(
                    device=model_input.device
                )

                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                prompt_embeds = prompt_embeds.to(
                    device=accelerator.device, dtype=weight_dtype
                )
                pooled_prompt_embeds = pooled_prompt_embeds.to(
                    device=accelerator.device, dtype=weight_dtype
                )
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=None,
                    return_dict=False,
                )[0]

                model_pred = model_pred * (-sigmas) + noisy_model_input

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=weighting_scheme, sigmas=sigmas
                )
                target = model_input
                loss = torch.mean(
                    (
                        weighting.float()
                        * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                avg_loss = accelerator.gather(
                    loss.repeat(config["per_gpu_batch_size"])
                ).mean()
                loss = loss.mean()
                train_loss += (
                    avg_loss.item() / config["gradient_accumulation_steps"]
                )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:            
                    accelerator.clip_grad_norm_(
                    transformer.parameters(), config["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

            if accelerator.sync_gradients:

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if accelerator.is_main_process:
                    if global_step % config["checkpointing_steps"] == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config["checkpoints_total_limit"] is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if (
                                len(checkpoints)
                                >= config["checkpoints_total_limit"]
                            ):
                                num_to_remove = (
                                    len(checkpoints)
                                    - config["checkpoints_total_limit"]
                                    + 1
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
                                        output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        transformer = accelerator.unwrap_model(transformer)
       
        if load_path != id:
            vae = None
        
        pipeline = StableDiffusion3Pipeline.from_pretrained(
                id,
                transformer=transformer,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                text_encoder_2=None,
                tokenizer_2=None,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch.float16,
            )
        if config["save_model"]:
            pipeline.save_pretrained(output_dir)
        pipeline = pipeline.to(accelerator.device)

        log_path = os.path.join(output_dir, "log.json")
        data = {"iteration": last_iteration + max_steps}
        with open(log_path, "w") as f:
            json.dump(data, f, indent=4)
    accelerator.end_training()
    return output_dir

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

    output_dir = train(train_config, caption_config)


    copy_file(args.config_train, output_dir)
    if args.config_caption != None: 
        copy_file(args.config_caption, output_dir)
        



