import sys

sys.path.append("./DiffusionSat")
import warnings

# Suppress all warnings
warnings.simplefilter("ignore")

from torch.utils.data import DataLoader, Dataset
import torch
import copy
import diffusers
from diffusionsat import (
    SatUNet, DiffusionSatPipeline,
    SampleEqually,
    fmow_tokenize_caption, fmow_numerical_metadata,
    spacenet_tokenize_caption, spacenet_numerical_metadata,
    satlas_tokenize_caption, satlas_numerical_metadata,
    combine_text_and_metadata, metadata_normalize,
)

from diffusers.optimization import get_scheduler

from tqdm import tqdm
import argparse
import logging
import os
import random
import json
import yaml
import torch
import shutil
import math
import numpy as np
from diffusers.optimization import get_scheduler
import torch.nn.functional as F

from diffusers import DDIMScheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
import random

from data_utils.datasets import Satellite_Dataset
from accelerate.utils import ProjectConfiguration, set_seed


def update_num_metadata(path, num_metadata):
    path = os.path.join(path, "config.json")
    try:
        # Open the JSON file for reading
        with open(path, "r") as file:
            data = json.load(file)

        # Update the num_metada value
        data["num_metadata"] = num_metadata

        # Open the JSON file for writing
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

        print("num_metadata value updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


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



def freeze_parameters_blocks(model):
    for name, layer in model.named_children():
        if "up_blocks" in name :
            layer.requires_grad_(True)
        else:
            layer.requires_grad_(False)
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

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    num_metadata = config['num_metadata']
    
    update_num_metadata(os.path.join(config['unet_path'], 'unet'), num_metadata)
    unet = SatUNet.from_pretrained(config['unet_path'] ,subfolder='unet',torch_dtype=torch.float32, low_cpu_mem_usage = False)
    # unet = SatUNet.from_pretrained(
    #     config['unet_path'] , subfolder="unet", 
    #     num_metadata=7, use_metadata=True, low_cpu_mem_usage=False,
    # )

    pipe = DiffusionSatPipeline.from_pretrained(config['pipeline_path'], unet=unet, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    noise_scheduler = DDIMScheduler.from_pretrained(config['pipeline_path'], subfolder="scheduler")
    output_dir = config['output_dir']


    accelerator_project_config = ProjectConfiguration(
            project_dir=config['project_dir'], logging_dir=os.path.join(config['logging_dir'], config['model_name'] + config['tracker_name']))
    accelerator = Accelerator(
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            mixed_precision="fp16",
            log_with=config['log_with'],
            project_config=accelerator_project_config,
        )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()


    generator = torch.Generator(
            device=accelerator.device).manual_seed(config['seed'])
    set_seed(config['seed'])


    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.decoder.to(accelerator.device, dtype=weight_dtype)
    pipe.unet.requires_grad_(True)
    parameters_list = [p for p in pipe.unet.parameters() if p.requires_grad]
    ################################
    ###############################
    

    optimizer_cls = torch.optim.AdamW
    learning_rate = (float(config['learning_rate']))
    optimizer = optimizer_cls(
            parameters_list,
            lr=learning_rate,
            betas=(config['adam_beta1'] , config['adam_beta2']),
            weight_decay=float(config['weight_decay']),
            eps=float(config['adam_epsilon'])
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
    pipe.unet, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        pipe.unet, optimizer, lr_scheduler, train_loader
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
        os.makedirs(output_dir,exist_ok=True)

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
    logger.info("***** Running training *****")
    logger.info(f"  Trainable params  = {sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)}")
    logger.info(f"  Num Epochs = {config['max_epochs']}")
    logger.info(
        f"  Instantaneous batch size per device = {config['per_gpu_batch_size']}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    last_iteration = 0

    if config['mode'] == 'train':
        for epoch in range(first_epoch, max_epochs + first_epoch):
                train_loss = 0.0
                
                pipe.unet.train()
                
                for step, batch in enumerate(train_loader):
                    
                    if config['resume_from_checkpoint'] != "" and epoch == first_epoch and step < resume_step:
                            if step % config['gradient_accumulation_steps'] == 0:
                                progress_bar.update(1)
                            continue 
                    with accelerator.accumulate(pipe.unet):
                        prompt = batch['caption_clip']
                        metadatas = batch['metadata'][:,0:num_metadata].to(weight_dtype).to(
                                accelerator.device, non_blocking=True
                                )
                        images = batch['img'].to(weight_dtype).to(
                                accelerator.device, non_blocking=True
                                )
                
                        prompt_embeds = pipe._encode_prompt(
                                prompt,
                                device,
                                1,
                                False,
                        )
                        latents = pipe.vae.encode(images).latent_dist.sample()
                        latents = latents * pipe.vae.config.scaling_factor 
                        
                        noise = torch.randn_like(latents)
                        noise = noise + config['noise_offset'] * torch.randn(
                                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                        
                        new_noise = noise + config['input_perturbation'] * torch.randn_like(noise)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        if config['input_perturbation']:
                            noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                        else:
                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        
                        # Get the target for loss depending on the prediction type
                        if config['prediction_type'] != 'default':
                            # set prediction_type of scheduler if defined
                            noise_scheduler.register_to_config(prediction_type=config['prediction_type'])

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        
                        input_metadata = pipe.prepare_metadata(bsz, metadatas, False, device, prompt_embeds.dtype)
                        model_pred = pipe.unet(
                            noisy_latents,
                            timesteps,
                            metadata=input_metadata,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False
                            )[0]
                        if config['loss'] == 'mse_loss':
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # if torch.isnan(loss):

                        #     logger.warning("Loss is NaN at step {global_step}. Skipping this step.")
                            
                        #     print(batch['metadata'][:,0:num_metadata])
                        #     print(batch['caption'])
                        #     print(batch['path'])
                        #     print("****************************************")
                            
                        #     continue

                        ######################
                        avg_loss = accelerator.gather(loss.repeat(config['per_gpu_batch_size'])).mean()
                        train_loss += avg_loss.item() / config['gradient_accumulation_steps']

                        # Backpropagate
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(pipe.unet.parameters(), config['max_grad_norm'])
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        progress_bar.update(1)
                    

                    if accelerator.sync_gradients:
                        
                        
                        

                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0
                        if accelerator.is_main_process:
                            if global_step % config['checkpointing_steps'] == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                if config['checkpoints_total_limit'] is not None:
                                    checkpoints = os.listdir(output_dir)
                                    checkpoints = [
                                        d for d in checkpoints if d.startswith("checkpoint")]
                                    checkpoints = sorted(
                                        checkpoints, key=lambda x: int(x.split("-")[1]))

                                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                    if len(checkpoints) >= config['checkpoints_total_limit']:
                                        num_to_remove = len(
                                            checkpoints) - config['checkpoints_total_limit'] + 1
                                        removing_checkpoints = checkpoints[0:num_to_remove]

                                        logger.info(
                                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                        )
                                        logger.info(
                                            f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                        for removing_checkpoint in removing_checkpoints:
                                            removing_checkpoint = os.path.join(
                                                output_dir, removing_checkpoint)
                                            shutil.rmtree(removing_checkpoint)

                                save_path = os.path.join(
                                    output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}") 

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(pipe.unet)
        pipeline  = DiffusionSatPipeline.from_pretrained(config['pipeline_path'], unet=accelerator.unwrap_model(pipe.unet), torch_dtype=torch.float16) 
        if config['save_model']:
            pipeline.save_pretrained(output_dir)
        pipeline = pipeline.to(accelerator.device)
    accelerator.end_training()


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
