import torch
from diffusers import StableDiffusion3Pipeline,SD3Transformer2DModel
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
from data_utils.datasets import Satellite_Dataset
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def read_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path, "r") as file:
            config = json.load(file)
    return config

def validate(model_name,rank=64, epoch =1, use_numerical_values=True, mask=None, input_size = 512, batch_size = 1, num_workers = 8):
    device = 'cuda'
    root_path = "/home/ecomapper/data/datasets"

    save_folder = os.path.join(root_path, 'logs', model_name, str(epoch),  str(input_size))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model_path = os.path.join(root_path, 'models', 'finetuned', model_name)
    transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path=model_path, subfolder="transformer", torch_dtype=torch.float16)

    print("loading weights...")
    pipe = StableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers", transformer = transformer,torch_dtype=torch.float16) 
       # if "official" not in model_name:
    #     
    #     safetensor_path = os.path.join(model_path, "transformer.safetensors")
    #     weights = load_file(safetensor_path, device=device) 

    #     transformer_lora_config = LoraConfig(
    #         r=rank,
    #         lora_alpha=rank,
    #         init_lora_weights="gaussian",
    #         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    #     )
    #     pipe.transformer.add_adapter(transformer_lora_config) 
    #     pipe.transformer.load_state_dict(weights, strict=True)
        
    pipe.to(device)
    
    
   

    csv_paths = ["/home/ecomapper/data/datasets/seed_98_final_test_single_image.csv"]
    dataset_names = ["proportional_sampled_points_seed_98_test"]
    
    dataset = Satellite_Dataset(csv_paths, dataset_names, root_path, use_numerical_values, mask, input_size, is_train=False)
    val_loader = DataLoader(dataset, batch_size= batch_size, num_workers = num_workers, pin_memory = True) 
    idx = 0
    batch_number = 0 
    for batch in val_loader:
        batch_number +=1

        
        prompt_short = batch['caption_clip']
        prompt_long = batch['caption_t5']
                                                                          
        images = pipe(

        prompt=prompt_short,
        prompt_2=prompt_short,
        prompt_3=prompt_long,
        num_inference_steps=50,
        guidance_scale=7.5,
        width =input_size,
        height = input_size

        
        ).images

        for i in range(len(images)):

            image_name = f'val_img_{idx}.jpg'         
            json_name = f'val_img_{idx}.json' 
            data = {'val_idx' : idx, 'path' : batch['img_path'][i], 'prompt_short':prompt_short[i], 'prompt_long':prompt_long[i] }
            json_str = json.dumps(data, indent=4)
            json_line = json_str.splitlines()
            # Split the JSON string into lines
            # Write each line to the file
            with open(os.path.join(save_folder, json_name), 'w') as json_file:
                for line in json_line:
                    json_file.write(line + '\n')
            images[i].save(os.path.join(save_folder, image_name))

            idx += 1


if __name__ == "__main__":
    root_json = "/home/ecomapper/Main/Ecomapper/configs/caption/"
    masks_names = [  
        'fixed_caption.json',
        'skip_month_always.json',
        'fixed_caption_word.json',
        'skip_weather_always.json'         
    ]


    
    mask_path = os.path.join(root_json, masks_names[0])
    model_name = "HPC_sd3_basemodel_large_new_captions_cleaned_2"
    #model_name = "Final_sd3_lora_512_rank_64_skip_weather_always_2_epochs"
    # mask_path = os.path.join(root_json, masks_names[1])
    # model_name = "Final_s__d3_lora_512_rank_64_skip_month_always"

    # mask_path = os.path.join(root_json, masks_names[0])
    # model_name = "Final_sd3_lora_512_rank_64_skip_loc_and_date"

    # mask_path = os.path.join(root_json, masks_names[0])
    # model_name = "Final_sd3_lora_512_rank_64_skip_all"

    json_file = read_config(mask_path)
    mask = json_file['masks']
    use_numerical_values = json_file['use_numerical_values']
    rank = 64
    validate(model_name, rank=rank,  epoch=2, mask=mask, use_numerical_values=use_numerical_values, input_size=512, batch_size=15, num_workers=16)
    #validate(model_name, rank=rank,  epoch=2, mask=mask, use_numerical_values=use_numerical_values, input_size=1024, batch_size=5, num_workers=16)


    

    


   
