import os
import matplotlib.pyplot as plt

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from peft import LoraConfig
from diffusers import StableDiffusion3Pipeline,SD3Transformer2DModel





def get_descriptive_word(value, data_type):
    """
    Given a value and its type (average_temp, average_prep, average_rad), 
    return a descriptive word based on predefined thresholds.
    """
    if data_type == 'average_temp':
        if value < -40:
            return 'Extremely Cold'
        elif -40 <= value < -30:
            return 'Very Cold'
        elif -30 <= value < -20:
            return 'Severely Cold'
        elif -20 <= value < -10:
            return 'Cold'
        elif -10 <= value < 0:
            return 'Chilly'
        elif 0 <= value < 10:
            return 'Cool'
        elif 10 <= value < 20:
            return 'Mild'
        elif 20 <= value < 30:
            return 'Warm'
        elif 30 <= value < 40:
            return 'Hot'
        else:
            return 'Very Hot'
    
    elif data_type == 'average_prep':
        epsilon = 1e-2
        if value <= epsilon:
            return 'No Precipitation'
        elif epsilon < value <= 1:
            return 'Very Light'
        elif 1 < value <= 5:
            return 'Light'
        elif 5 < value <= 10:
            return 'Moderately Light'
        elif 10 < value <= 20:
            return 'Moderate'
        elif 20 < value <= 30:
            return 'Moderately Heavy'
        elif 30 < value <= 50:
            return 'Heavy'
        elif 50 < value <= 75:
            return 'Very Heavy'
        else:
            return 'Extreme'
    
    elif data_type == 'average_rad':
        if value < 2:
            return 'Extremely Low'
        elif 2 <= value < 5:
            return 'Very Low'
        elif 5 <= value < 10:
            return 'Low'
        elif 10 <= value < 15:
            return 'Moderate'
        elif 15 <= value < 20:
            return 'Somewhat High'
        elif 20 <= value < 25:
            return 'High'
        elif 25 <= value < 30:
            return 'Very High'
        elif 30 <= value < 40:
            return 'Extreme'
        else:
            return 'Ultra Extreme'
    
    return 'Unknown'

def create_prompt(location, land_type, month, temp, prep, rad, use_numerical_values = True, drop_month = False, drop_loc=False):
    loc_info = f" in {location}"
    month_info = f" {month}"
    if drop_month:
        month_info = ""

    if drop_loc:
        loc_info = ""

    

    prompt_short = f"a satellite image of {land_type}{loc_info} on{month_info} 2019."

    if use_numerical_values:
        
        prompt_long = prompt_short + f" The average temperature over the last month was {temp} C. with an average precipitation of {prep} mm, and an average daily solar radiation of {rad} W/m2."

    else:
        temp = get_descriptive_word(temp, "average_temp")
        prep = get_descriptive_word(prep, "average_prep") 
        rad = get_descriptive_word(rad, "average_rad") 
        prompt_long = prompt_short + f" The average temperature over the last month was {temp}, with {prep} precipitation, and {rad} daily solar radiation."

    return prompt_short, prompt_long

def generate_images(pipe, latents, prompt_short, prompt_long, width=512, height=512):
    # Generate the image using the updated model pipeline with additional parameters
    

        

    num_inference_steps = 50
    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # timesteps = pipe.scheduler.timesteps

    image = pipe(
        prompt=prompt_short,
        prompt_2=prompt_short,
        prompt_3=prompt_long,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        latents = latents,
        #timesteps = timesteps

    ).images[0]
    return image

def save_images_in_grid(images, month, location, land_type, root_path):
    # Create a folder structure for the experiment
    folder_path = os.path.join(root_path, location, land_type)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create a grid figure for the 45 images for this month (5 temp * 9 (precip * radiation))
    fig, axs = plt.subplots(1, 3, figsize=(25, 15))  # 5 rows for temperature, 9 columns for precip/radiation
    axs = axs.flatten()  # Flatten the grid for easy indexing
    
    # Iterate through each image and set it on the grid with labels
    for i, ax in enumerate(axs):
        ax.imshow(images[i][0])  # Get image from the list
        ax.axis('off')
        
        # Extract temperature, precipitation, and radiation values for the title
        temp = images[i][1]  # Get the temperature from the list
        precip = images[i][2]  # Get the precipitation from the list
        rad = images[i][3]  # Get the radiation from the list
        
        # Set title under each image with numerical values
        ax.set_title(f'Temperature: {temp}°C\nPrecip: {precip}mm\nRadiation: {rad}W/m²', fontsize=8, pad=10)
    
    # Save the figure as a PNG
    file_path = os.path.join(folder_path, f"{month}_experiment.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def run_experiment(pipe, root_path, use_numerical_values ,drop_month, drop_location):
    extreme_values = {
        # 'temperature': [-40, -20, 0, 20, 40],   # extreme cold to extreme hot
        'temperature' : [-10, 20, 40 ],
        'precipitation': [0, 20, 50],          # dry to heavy precipitation
        'radiation': [1, 10, 30],            # low to high solar radiation
        'months': ['January', 'July', 'December', 'March', 'August'],  # months with extreme changes
        'locations': [
            # "Beijing, China",
             "Moscow, Russia",
             "Cape Town, South Africa",
          
            "Chicago, United States",
             "Ontario, Canada",
            "Sydney, Australia",
            "Germany",
            "Hamburg, Germany",
            "Munich, Germany"

            # "Buenos Aires, Argentina",
            # "Dubai, United Arab Emirates"
        ],
        'land_types': [
            "Evergreen Needleleaf Forests",
            "Barren or Sparsely Vegetated",
            "Croplands",
            "Grasslands",
            "Open Shrublands",
            "Savannas",
            "Woody Savannas",
            "Evergreen Broadleaf Forests",
        ]
    }
    batch_size = 1
    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        batch_size,
        num_channels_latents,
        1024 // pipe.vae_scale_factor,
        1024 // pipe.vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.float16)

    if drop_month:
        extreme_values['months'] = ['January']

    if drop_location:
        extreme_values['locations'] = ["Beijing, China"]
    

    
    for location in extreme_values['locations']:
        for land_type in extreme_values['land_types']:
            for month in extreme_values['months']:
                
                # Prepare the list of images for the current month
                images = []
                
                for temp in extreme_values['temperature']:
                    for precip in extreme_values['precipitation']:
                        for rad in extreme_values['radiation']:
                            if (temp == -10 and precip == 0 and rad == 1) or ( temp == 20 and  precip == 20 and rad== 10)  or (temp == 40 and precip == 50 and rad == 30):
                                # Create the prompt and generate the image
                                prompt_short, prompt_long = create_prompt(location, land_type, month, temp, precip, rad, use_numerical_values, drop_month, drop_location)
                                image = generate_images(pipe, latents, prompt_short, prompt_long)
                                
                                # Append the image and its corresponding values to the list
                                images.append([image, temp, precip, rad])
                
                # Save the generated images in a grid
                save_images_in_grid(images, month, location, land_type, root_path)


def validate(model_name, use_numerical_values = True, drop_month= False, drop_location = False, seed=44):

    


    root_path = "/home/ecomapper/data/datasets/experiments_with_fixed_noise/"
    
    
    num = "with_numerical"
    if not use_numerical_values:
        num = "with_words"
    month = "with_month"
    if drop_month:
        month = "drop_month"

    loc = "with_location"
    if drop_location:
        loc = "drop_location"

    save_folder = os.path.join(root_path, model_name, str(seed), num, month, loc)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    root_path = "/home/ecomapper/data/datasets"
    model_path = os.path.join(root_path, 'models', 'finetuned', model_name)
    transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path=model_path, subfolder="transformer", torch_dtype=torch.float16)

    print("loading weights...")
    pipe = StableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path = "stabilityai/stable-diffusion-3-medium-diffusers", transformer = transformer,torch_dtype=torch.float16) 
    # print("loading weights...")
    # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    # root_path = "/home/ecomapper/data/datasets"
    # model_path = os.path.join(root_path, 'models', 'finetuned', model_name)
    # safetensor_path = os.path.join(model_path, "transformer.safetensors")
    # weights = load_file(safetensor_path, device=device) 

    # rank = 64
    # transformer_lora_config = LoraConfig(
    #     r=rank,
    #     lora_alpha=rank,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # pipe.transformer.add_adapter(transformer_lora_config) 
    # pipe.transformer.load_state_dict(weights, strict=True)
    
    pipe.to(device)

    run_experiment(pipe, save_folder, use_numerical_values, drop_month, drop_location)

if __name__ == "__main__":
    
    #model_name = "Final_sd3_lora_512_rank_64_skip_month_always"
    #validate(model_name, use_numerical_values=True, drop_month=True)
    seeds = [333, 666, 999]
    device = 'cuda'

    for seed in seeds:
    
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model_name = "HPC_sd3_basemodel_large_new_captions_cleaned_2"
        #model_name = "Final_sd3_lora_512_rank_64_skip_all"
        #model_name = "Final_sd3_lora_512_rank_64_skip_loc_and_date_word"
        validate(model_name, use_numerical_values=True, drop_month=True, drop_location=False, seed= seed)
        #validate(model_name, use_numerical_values=True, drop_month=True, drop_location=False, seed=seed)
        #validate(model_name, use_numerical_values=True, drop_month=False, drop_location=False, seed=seed)
        #validate(model_name, use_numerical_values=True, drop_month=False, drop_location=True)
    

