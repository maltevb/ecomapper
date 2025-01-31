import sys

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

)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
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
from safetensors.torch import save_file, load_file,load_model

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

def train(config, config_caption):   
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    id = "stabilityai/stable-diffusion-3-medium-diffusers"
    output_dir = os.path.join(config["output_dir"], config["model_name"])

    
    vae = AutoencoderKL.from_pretrained(
        id, subfolder="vae", torch_dtype=torch.float16
    ) 
    transformer = SD3Transformer2DModel.from_pretrained(
        id, subfolder="transformer", torch_dtype=torch.float16
    )
    vae = vae.to("cuda")
    transformer = transformer.to("cuda")
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        id, subfolder="scheduler"
    )


    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
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

    generator = torch.Generator(device=accelerator.device).manual_seed(config["seed"])
    set_seed(config["seed"])

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    weight_dtype = torch.float32
    vae.to(accelerator.device, dtype=torch.float32)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16



    ###################### HPC #########################
   
    pipe = StableDiffusion3Pipeline.from_pretrained(
        id, transformer=None, vae=None, torch_dtype=torch.float16
    )
    pipe.text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_3.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    

    rank = config["rank"]
    #######################################################
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    if config['load_path']:
        path = os.path.join(config['load_path'] ,"transformer.safetensors")
        missing = load_model(transformer, path, device = accelerator.device)

    transformer.to(accelerator.device, dtype=weight_dtype)


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    all_layers = model.state_dict()
                        
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            save_file(all_layers, os.path.join(output_dir,"transformer.safetensors"))

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        # lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        # transformer_state_dict = {
        #     f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
        # }
        # transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

        # incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        # if incompatible_keys is not None:
        #     # check only for unexpected keys
        #     unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        #     if unexpected_keys:
        #         logger.warning(
        #             f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
        #             f" {unexpected_keys}. "
                

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            path = os.path.join(input_dir, "transformer.safetensors")
            missing = load_model(transformer_, path, device = "cuda")
            

        transformer_.to(accelerator.device, dtype=weight_dtype)

        if  accelerator.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    

    ################################
    ###############################

    
    learning_rate = float(config["learning_rate"])
    if config['input_size'] == 512:
        transformer.norm_out.requires_grad_(True)
        transformer.proj_out.requires_grad_(True)
    
    if accelerator.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": learning_rate}
    parameters_list = [transformer_parameters_with_lr]


    if config['use_8bit_adam']:
        optimizer_cls = bnb.optim.AdamW8bit  # torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        parameters_list,
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_epsilon"]),
    )

    ###################### DATA LOADER ###############################
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

    
   ##################################################################

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
   
    logger.info(
            f"  Trainable params  = {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
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
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    max_epoch = max_epochs + first_epoch
    #####

    if config["mode"] == "train":
        for epoch in range(first_epoch, max_epoch):

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
                with accelerator.accumulate(
                    transformer
                ):

                    pixel_values = batch["img"].to(dtype=vae.dtype).to(accelerator.device, non_blocking=True)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor         
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

                    prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)

           

                   
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
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
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

                    # Get the target for loss depending on the prediction type
                    
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                    model_pred = model_pred * (-sigmas) + noisy_model_input

                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(
                        weighting_scheme=weighting_scheme, sigmas=sigmas
                    )

                    # flow matching loss
                    target = model_input

                    # Compute regular loss.
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
                        params_to_clip = transformer_parameters
                        accelerator.clip_grad_norm_(params_to_clip, config["max_grad_norm"])
                       
                   
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
        transformer = unwrap_model(transformer)
        transformer = transformer.to(torch.float32)
        all_layers = transformer.state_dict()


        ##



        ##
        save_file(all_layers, os.path.join(output_dir,"transformer.safetensors"))

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
        
