import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3ControlNetPipeline, SD3ControlNetModel
import sys
import random
import numpy as np
import os
import argparse
import json
from safetensors.torch import load_file, load_model
from transformers import AutoModel
from transformers import AutoConfig
from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
seed = 7777
from PIL import Image
import torchvision.transforms as transforms
sys.path.append("..")
from data_utils.datasets import Sequence_Satellite_Dataset
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def validate(transformer_model_name, controlnet_model_name, masks, epoch =1, use_numerical_values=True, input_size = 512, batch_size = 1, num_workers = 8):
    device = 'cuda'
    root_path = "/home/ecomapper/data/datasets"

    save_folder = os.path.join(root_path, 'logs', "controlnet", transformer_model_name, str(epoch),  str(input_size))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    print("Loading transformer weights...")
    transformer = SD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer", torch_dtype=torch.float16
    )
    model_path = os.path.join(root_path, 'models', 'finetuned', transformer_model_name)
    safetensor_path = os.path.join(model_path, "transformer.safetensors")
    weights = load_file(safetensor_path, device=device) 
    print(f"Transformer weights loaded. Type: {type(weights)}")
    
    rank = 64
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config) 

    transformer.load_state_dict(weights, strict=True)
    
    print("Transformer weights are loaded.")
    
    print("Loading controlnet weights...")
    
    controlnet = SD3ControlNetModel.from_transformer(
        transformer,
        num_layers=12,
        num_extra_conditioning_channels=0,
    )
    
    controlnet_model_path = os.path.join(root_path, 'models', 'finetuned', controlnet_model_name)
    safetensor_path = os.path.join(controlnet_model_path, "controlnet.safetensors")
    weights = load_file(safetensor_path, device=device) 
    print(f"ControlNet weights loaded. Type: {type(weights)}")
    controlnet.add_adapter(transformer_lora_config) 
    
    controlnet.load_state_dict(weights, strict=False)
            
    controlnet.to(torch.float16)
    
    print("Controlnet weights are loaded.")
    
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    csv_paths = ["/home/ecomapper/data/datasets/seed_98_final_test_two_images_future.csv"]
    dataset_names = ["proportional_sampled_points_seed_98_test"]
    is_train = False
    
    print("Loading validation dataset.")
    
    dataset = Sequence_Satellite_Dataset(
        csv_paths=csv_paths,
        dataset_names=dataset_names, 
        root=root_path,
        past_month_max=6,
        use_numerical_values=use_numerical_values,
        masks=masks,
        img_size=input_size,
        is_train=is_train
    )
    print(len(dataset))
    val_loader = DataLoader(dataset, batch_size= batch_size, num_workers = num_workers, pin_memory = True) 
    print("Validation dataset is loaded.")

    idx = 0
    print(len(val_loader))
    for batch in val_loader:
        
        control_image = batch["past_img"]
        prompt_short = batch['caption_clip']
        prompt_long = batch['caption_t5']

        
        for i in range(len(control_image)):
            control_image_input = control_image[i].unsqueeze(0).to(torch.float16)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                images = pipe(
                    prompt=prompt_short[i],
                    prompt_2=prompt_short[i],
                    prompt_3=prompt_long[i],
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    width =input_size,
                    height = input_size,
                    control_image=control_image_input
                ).images[0]

            image_name = f'val_img_{idx}.jpg'         
            json_name = f'val_img_{idx}.json' 
            data = {
                'val_idx' : idx, 
                'gt_path' : batch['img_path'][i], 
                'past_img_path' : batch['past_img_path'][i], 
                'prompt_short':prompt_short[i], 
                'prompt_long':prompt_long[i] 
            }
            json_str = json.dumps(data, indent=4)
            json_line = json_str.splitlines()
            # Split the JSON string into lines
            # Write each line to the file
            with open(os.path.join(save_folder, json_name), 'w') as json_file:
                for line in json_line:
                    json_file.write(line + '\n')
            images.save(os.path.join(save_folder, image_name))

            idx += 1


def read_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path, "r") as file:
            config = json.load(file)
    return config

if __name__ == "__main__":

    transformer_model_names = ['Final_controlnet_lora_512_rank_64_skip_loc_and_date_2_epochs_finetune_12_all_1_epoch_w_2seeds']
    controlnet_model_names = ['Final_controlnet_lora_512_rank_64_skip_loc_and_date_2_epochs_finetune_12_all_1_epoch']
    
    input_sizes = [512]
    epochs = [1]
    # use_numerical_values = [True] 
    batch_size = 10
    root_json = "/home/ecomapper/Main/Ecomapper/configs/caption/"
    mask_path = os.path.join(root_json, "skip_date_location.json")
    json_file = read_config(mask_path)
    mask = json_file['masks']
    use_numerical_values = json_file['use_numerical_values']
    
    for i in range(len(transformer_model_names)):
        validate(
            transformer_model_name = transformer_model_names[i], 
            controlnet_model_name = controlnet_model_names[i], 
            epoch = epochs[i], 
            use_numerical_values = use_numerical_values, 
            input_size = input_sizes[i], 
            batch_size = batch_size, 
            masks = mask
        )



    


   
