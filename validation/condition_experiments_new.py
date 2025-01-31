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
# from data_utils.datasets import Sequence_Satellite_Dataset
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import pandas as pd
import calendar
from datetime import datetime
from diffusers.utils.torch_utils import randn_tensor

def convert_cloud_info(cloud_cover):
        if cloud_cover > 0.7:
            return "very cloudy "
        elif 0.5 < cloud_cover <= 0.7:
            return "cloudy "
        elif 0.3 < cloud_cover <= 0.5:
            return "partially cloudy "
        else:
            return ""
    
def convert_date_string(date_str):
    # Split the input string to extract year and month
    year, month = map(int, date_str.split("-"))

    # Get the full month name from the month number
    month_name = calendar.month_name[month]

    # Return the formatted string
    return month_name, year

def convert_location(location):
    if ", " in location:
        state, country = location.split(", ", 1)  # Split only at the first occurrence of ", "
    else:
        state, country = "", location  # If there's no comma, set state as empty and country as the full string
    
    return state, country

def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    # Here: keep them as a "list of lists of strings"
    base_captions = [item["base_caption"] for item in batch]
    current_full_captions = [item["current_full_caption"] for item in batch]
    img_paths = [item["img_path"] for item in batch]
    future_base_captions = [item["future_base_captions"] for item in batch]
    future_full_captions = [item["future_full_captions"] for item in batch]
    location_id = [int(item["location_id"]) for item in batch]
    image_name = [item["image_name"] for item in batch]
    # If you do want images as a batch tensor:
    images = torch.stack(images, dim=0)
    return {
        "image": images,
        "future_base_captions": future_base_captions,
        "future_full_captions": future_full_captions,
        "location_id": location_id,
        "image_name": image_name,
        "img_path": img_paths,
        "base_caption": base_captions,
        "current_full_caption": current_full_captions
        
    }
    
class CaptionBuilder:
    def __init__(self, use_numerical_values=False):
        self.use_numerical_values = use_numerical_values
    
    


    def generate(self, row):
        # Current metadata
        land_type = row["type"]
        month, year = convert_date_string(row["date"])
        cloud_info = convert_cloud_info(row["cloud_coverage"])
        state, country = convert_location(row['location_address'])

        if self.use_numerical_values:
            temp = round(row['average_temp'])
            prep = round(row['average_prep'])
            rad = round(row['average_rad'])

        if self.use_numerical_values:
            temp_2050 = round(row.get('2050_temp', row['average_temp']))
            prep_2050 = round(row.get('2050_pr', row['average_prep']))
            rad_2050 = round(row.get('2050_rsds', row['average_rad']))

            temp_2075 = round(row.get('2075_temp', row['average_temp']))
            prep_2075 = round(row.get('2075_pr', row['average_prep']))
            rad_2075 = round(row.get('2075_rsds', row['average_rad']))

            temp_2100 = round(row.get('2100_temp', row['average_temp']))
            prep_2100 = round(row.get('2100_pr', row['average_prep']))
            rad_2100 = round(row.get('2100_rsds', row['average_rad']))

        # Base caption
        base_caption = f"A {cloud_info}satellite image of {land_type} in {state}, {country} on {month} {year}."

        # Current climate caption
        current_climate_caption = (
            f" The average temperature over the last month was {temp} C. with an average precipitation {prep} mm, and an average daily solar radiation of {rad} W/m2."
        )
        
        base_caption_2050 = f"A {cloud_info}satellite image of {land_type} in {state}, {country} on {month} 2050."
        climate_caption_2050 = (
            f" The average temperature over the last month was {temp_2050} C. with an average precipitation {prep_2050} mm, and an average daily solar radiation of {rad_2050} W/m2."
        )
        
        base_caption_2075 = f"A {cloud_info}satellite image of {land_type} in {state}, {country} on {month} 2075."
        climate_caption_2075 = (
            f" The average temperature over the last month was {temp_2075} C. with an average precipitation {prep_2075} mm, and an average daily solar radiation of {rad_2075} W/m2."
        )
        
        base_caption_2100 = f"A {cloud_info}satellite image of {land_type} in {state}, {country} on {month} 2100."
        climate_caption_2100 = (
            f" The average temperature over the last month was {temp_2100} C. with an average precipitation {prep_2100} mm, and an average daily solar radiation of {rad_2100} W/m2."
        )
        
        future_base_captions = [base_caption_2050, base_caption_2075, base_caption_2100]
        full_captions_2050 = base_caption_2050 + climate_caption_2050
        full_captions_2075 = base_caption_2075 + climate_caption_2075
        full_captions_2100 = base_caption_2100 + climate_caption_2100
        future_full_captions = [full_captions_2050, full_captions_2075, full_captions_2100]

        # Full caption
        current_full_caption = base_caption + current_climate_caption

        return base_caption, current_full_caption, future_base_captions, future_full_captions

class CustomSatelliteDataset(Dataset):
    def __init__(self, df, root_dir, img_size=512, use_numerical_values=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            root_dir (str): Root directory where images are stored.
            img_size (int): Size to resize the images.
            use_numerical_values (bool): Whether to use numerical values in captions.
        """
        self.df = df
        self.root_dir = root_dir
        self.img_size = img_size
        self.use_numerical_values = use_numerical_values
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row corresponding to the index
        row = self.df.iloc[idx]

        # Load the image
        img_path = os.path.join(self.root_dir, str(row['location_id']), row['img_path'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Generate captions for current and future metadata
        caption_builder = CaptionBuilder(use_numerical_values=self.use_numerical_values)
        base_caption, current_full_caption, future_base_captions, future_full_captions = caption_builder.generate(row)

        # Return the image, captions, and metadata
        return {
            "image": image,
            "base_caption": base_caption,
            "current_full_caption": current_full_caption,
            "img_path": img_path,
            "future_base_captions": future_base_captions,
            "future_full_captions": future_full_captions,
            "location_id": row['location_id'],
            "image_name" : row['img_path']
        }


def validate(transformer_model_name, controlnet_model_name, dataloader, masks, epoch =1, use_numerical_values=True, input_size = 512, batch_size = 1, num_workers = 8):
    device = 'cuda'
    root_path = "/home/ecomapper/data/datasets"

    save_folder = os.path.join(root_path, 'logs', "controlnet", controlnet_model_name, str(epoch),  "experiments_with_fixed_noise_filtered_countries_rainforests_april")

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

    # csv_paths = ["/home/ecomapper/data/datasets/seed_98_final_test_two_images_future.csv"]
    # dataset_names = ["proportional_sampled_points_seed_98_test"]
    # masks = None
    # is_train = False
    
    # print("Loading validation dataset.")
    
    # dataset = Sequence_Satellite_Dataset(
    #     csv_paths=csv_paths,
    #     dataset_names=dataset_names, 
    #     root=os.path.join(root_path, "datasets"),
    #     past_month_max=6,
    #     use_numerical_values=use_numerical_values,
    #     masks=masks,
    #     img_size=input_size,
    #     is_train=is_train
    # )
    # print(len(dataset))
    # val_loader = DataLoader(dataset, batch_size= batch_size, num_workers = num_workers, pin_memory = True) 
    # print("Validation dataset is loaded.")

    idx = 0
    print(len(dataloader))
    for batch in dataloader:
        
        control_image = batch["image"]
        prompt_short = batch['future_base_captions']
        prompt_long = batch['future_full_captions']
        # years = batch['future_years']
        location_id = batch['location_id']
        image_names = batch['image_name']
        future_years = [2050, 2075, 2100]
        
        for i in range(len(control_image)):
            # control_image = [img.half() for img in control_image]
            control_image_input = control_image[i].unsqueeze(0).to(torch.float16)
            # print("control_image_input dtype:", control_image_input.dtype)
            # print("controlnet weights dtype:", next(controlnet.parameters()).dtype)
            # print(batch)
            batch_size = 1
            num_channels_latents = pipe.transformer.config.in_channels
            shape = (
                batch_size,
                num_channels_latents,
                512 // pipe.vae_scale_factor,
                512 // pipe.vae_scale_factor,
            )
            latents = randn_tensor(shape, generator=None, device=device, dtype=torch.float16)
            for j in range(3):
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    images = pipe(
                        prompt=prompt_short[i][j],
                        prompt_2=prompt_short[i][j],
                        prompt_3=prompt_long[i][j],
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        width =input_size,
                        height = input_size,
                        control_image=control_image_input,
                        latents = latents
                    ).images[0]
                # replace previous year with future year in image name 3734_2018-07.png to 3734_2050-07.jpg
                year_value = int(future_years[j])
                loc_id_value = int(location_id[i])
                old_year_and_month = image_names[i].split("_")[1]         # "2019-11.png"
                month_part = old_year_and_month.split("-")[1].split(".")[0]  # "11"
                image_name = f"{loc_id_value}_{year_value}-{month_part}.jpg"
                json_name = image_name.replace(".jpg", ".json")
                data = {
                    'location_id' : loc_id_value, 
                    # 'gt_path' : batch['img_path'][i], 
                    'past_img_path' : batch['img_path'][i], 
                    'prompt_short':prompt_short[i][j], 
                    'prompt_long':prompt_long[i][j] 
                }
                json_str = json.dumps(data, indent=4)
                json_line = json_str.splitlines()
                # Split the JSON string into lines
                # Write each line to the file
                directory_path = os.path.join(save_folder, str(loc_id_value))
                os.makedirs(directory_path, exist_ok=True)
                with open(os.path.join(save_folder, str(loc_id_value),  json_name), 'w') as json_file:
                    for line in json_line:
                        json_file.write(line + '\n')
                images.save(os.path.join(save_folder, str(loc_id_value), image_name))

                idx += 1


def read_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path, "r") as file:
            config = json.load(file)
    return config


if __name__ == "__main__":
    root_path = "/home/ecomapper/data/datasets"
    # Path to your CSV file
    # csv_path = "/home/ecomapper/data/datasets/seed_98_final_test_two_images_future.csv"
    csv_path = "/home/ecomapper/data/datasets/filtered_countries_rainforests_april.csv"
    
    
    dataset_name = "proportional_sampled_points_seed_98_test"
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    print(df.head())
    root_dir = os.path.join(root_path, dataset_name)
        # Create the dataset
    dataset = CustomSatelliteDataset(df, root_dir, img_size=512, use_numerical_values=True)

    # Create the dataloader
    batch_size = 8
    num_workers = 4  # Adjust based on your system
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)

    # # Example: Iterate through the dataloader
    # for batch in dataloader:
    #     images = batch["image"]
    #     base_captions = batch["base_caption"]
    #     current_full_caption = batch["current_full_caption"]
    #     # metadata = batch["metadata"]
    #     img_paths = batch["img_path"]
    #     future_base_captions = batch["future_base_captions"]
    #     future_full_captions = batch["future_full_captions"]
    #     # future_years = batch["future_years"]
    #     location_id = batch["location_id"]
    #     image_name = batch["image_name"]

    #     print("Images shape:", images.shape)
    #     print("ımage length:", len(images))
    #     print("Base Captions:", base_captions)
    #     print("Current Full Captions:", current_full_caption)
    #     # print("Metadata:", metadata)
    #     print("Image Paths:", img_paths)
    #     print("Future Base Captions:", future_base_captions)
    #     print("Future Full Captions:", future_full_captions)
    #     # print("Future Years:", future_years)
    #     print("location_id",location_id)
    #     print("image_name",image_name)
    #     break  # Just for demonstration
    
    transformer_model_name = "Final_sd3_lora_512_rank_64_skip_loc_and_date_2_epochs"
    controlnet_model_name = "Final_controlnet_lora_512_rank_64_skip_loc_and_date_2_epochs_finetune_12_all_1_epoch"
    input_size = 512
    epoch = 1
    root_json = "/home/ecomapper/Main/Ecomapper/configs/caption/"
    mask_path = os.path.join(root_json, "skip_date_location.json")
    json_file = read_config(mask_path)
    mask = json_file['masks']
    use_numerical_values = json_file['use_numerical_values']
    
    validate(
            transformer_model_name = transformer_model_name, 
            controlnet_model_name = controlnet_model_name, 
            epoch = epoch, 
            use_numerical_values = use_numerical_values, 
            input_size = input_size, 
            batch_size = batch_size, 
            masks = mask,
            dataloader = dataloader
        )
    