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

def validate(transformer_model_name, controlnet_model_name, epoch =1, use_numerical_values=True, input_size = 512, batch_size = 1, num_workers = 8):
    device = 'cuda'
    root_path = "/home/ecomapper/data/datasets"

    save_folder = os.path.join(root_path, 'logs', "controlnet", controlnet_model_name, str(epoch),  str(input_size))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    print("Loading transformer weights...")
    if transformer_model_name == 'stabilityai/stable-diffusion-3-medium-diffusers':
        
        model_path = 'stabilityai/stable-diffusion-3-medium-diffusers'
    else:
        model_path = os.path.join(root_path, 'models', 'finetuned', transformer_model_name)
        
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float16
    )
    print("Transformer weights are loaded.")
    
    print("Loading controlnet weights...")
    controlnet_model_path = os.path.join(root_path, 'models', 'finetuned', controlnet_model_name)
    controlnet = SD3ControlNetModel.from_pretrained(
        pretrained_model_name_or_path=controlnet_model_path,
        torch_dtype=torch.float16,
    )
    
    
    controlnet.to(torch.float16)
    
    print("Controlnet weights are loaded.")
    
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    csv_paths = ["/home/ecomapper/data/datasets/seed_70_validation_sequence.csv"]
    dataset_names = ["proportional_sampled_points_seed_70"]
    masks = None
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


if __name__ == "__main__":

    transformer_model_names = ['stabilityai/stable-diffusion-3-medium-diffusers']
    controlnet_model_names = ['Final_controlnet_512_12_layer_original_transformer_skip_date_location_2_epoch']
    
    input_sizes = [512]
    epochs = [1]
    use_numerical_values = [True] 
    batch_size = 10

    for i in range(len(transformer_model_names)):
        validate(transformer_model_names[i], controlnet_model_names[i], epochs[i], use_numerical_values[i], input_sizes[i], batch_size)



    


   
