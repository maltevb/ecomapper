import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor


device = 'cuda'

# Define extreme weather values and months
extreme_values = {
    'temperature': [-50, -20, 0, 25, 50],   # extreme cold to extreme hot
    'precipitation': [0, 50, 100],          # dry to heavy precipitation
    'radiation': [1, 10, 35],            # low to high solar radiation
    'months': ['January', 'July', 'December', 'March', 'August'],  # months with extreme changes
    'locations': [
        "Beijing, China",
        "Moscow, Russia",
        "Cape Town, South Africa",
        "Mumbai, India",
        # "Mexico City, Mexico",
        # "Jakarta, Indonesia",
        # "Buenos Aires, Argentina",
        # "Dubai, United Arab Emirates"
    ],
    'land_types': [
        "Evergreen Needleleaf Forests",
        "Barren or Sparsely Vegetated",
        "Croplands",
        "Grasslands",
        "Open Shrublands",
        "Mixed Forests",
        "Evergreen Broadleaf Forests",
        "Closed Shrublands"
    ]
}



def create_prompt(location, land_type, month, temp, precip, rad):
    prompt_short = f"a  satellite image of {land_type} in {location} on {month} 2019."
    prompt_long = f"a  satellite image of {land_type} in {location} on {month} 2019. The average temperature over the last month was {temp} C. with an average precipitation of {precip} mm, and an average daily solar radiation of {rad} W/m2."
    return prompt_short, prompt_long

def generate_images(latents, prompt_short, prompt_long, width=1024, height=1024):
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

    ).images[0]
    return image

def save_images_in_grid(images, month, location, land_type, root_path):
    # Create a folder structure for the experiment
    folder_path = os.path.join(root_path, location, land_type)
    os.makedirs(folder_path, exist_ok=True)
    
    # Create a grid figure for the 45 images for this month (5 temp * 9 (precip * radiation))
    fig, axs = plt.subplots(5, 9, figsize=(25, 15))  # 5 rows for temperature, 9 columns for precip/radiation
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

def run_experiment(root_path):

    batch_size = 1
    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        batch_size,
        num_channels_latents,
        int(1024) // pipe.vae_scale_factor,
        int(1024) // pipe.vae_scale_factor,
    )

    
    latents = randn_tensor(shape, generator=None, device=device, dtype=torch.float16)


    for location in extreme_values['locations']:
        for land_type in extreme_values['land_types']:
            for month in extreme_values['months']:
                # Prepare the list of images for the current month
                images = []
                
                for temp in extreme_values['temperature']:
                    for precip in extreme_values['precipitation']:
                        for rad in extreme_values['radiation']:
                            # Create the prompt and generate the image
                            prompt_short, prompt_long = create_prompt(location, land_type, month, temp, precip, rad)
                            image = generate_images(latents, prompt_short, prompt_long)
                            
                            # Append the image and its corresponding values to the list
                            images.append([image, temp, precip, rad])
                
                # Save the generated images in a grid
                save_images_in_grid(images, month, location, land_type, root_path)




# Set your root path where all the experiments will be saved
root_path = "/home/ecomapper/data/datasets/experiments_with_fixed_noise/"
hf_token = ''
pipe = StableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, token=hf_token)
pipe = pipe.to(device)
model_root = '/home/ecomapper/data/datasets/models/finetuned'

model_name = '2025_sd3_lora_rank_64_1_epoch'
model_path = os.path.join(model_root, model_name)

experiment_folder = os.path.join(root_path, model_name)
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder, exist_ok = True)
print("loading lora")
pipe.load_lora_weights(model_path)
# Run the experiment generation
run_experiment(experiment_folder)
