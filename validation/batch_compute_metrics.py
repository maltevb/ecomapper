import os
import json
from PIL import Image
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips



import argparse
import json
import warnings
from tqdm import tqdm
# Ignore a specific warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load images in batches
def load_images_in_batches(img_path, batch_size=100):
    image_paths = [f for f in os.listdir(img_path) if f.endswith('.png') or f.endswith('.jpg')]
    jsons = [f for f in os.listdir(img_path) if f.endswith('.json')]

    image_paths.sort()
    jsons.sort()

    for i in range(0, len(image_paths), batch_size):
        imgs = []
        for path in image_paths[i:i + batch_size]:
            with Image.open(os.path.join(img_path, path)) as img:
                img = img.resize((512, 512))
                img_array = np.array(img) / 255.0
                imgs.append(img_array)

        gen_images_array = np.stack(imgs)
        
        ref_images_path = []
        for file in jsons[i:i + batch_size]:
            with open(os.path.join(img_path, file), 'r') as f:
                data = json.load(f)
                ref_img_path = data['path']
                ref_images_path.append(ref_img_path)
                
        ref_imgs = []
        for path in ref_images_path:
            with Image.open(path) as img:
                img = img.resize((512, 512))
                img_array = np.array(img) / 255.0
                ref_imgs.append(img_array)
        
        ref_images_array = np.stack(ref_imgs)
        
        yield gen_images_array, ref_images_array

# Function to calculate CLIP score in batches
def calculate_clip_in_batches(img_path, batch_size=100):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    total_score = 0
    count = 0
    
    image_paths = [f for f in os.listdir(img_path) if f.endswith('.png') or f.endswith('.jpg')]
    jsons = [f for f in os.listdir(img_path) if f.endswith('.json')]

    image_paths.sort()
    jsons.sort()
    
    for i in range(0, len(image_paths), batch_size):
        batch_images = []
        batch_prompts = []
        
        for img_file, json_file in zip(image_paths[i:i + batch_size], jsons[i:i + batch_size]):
            # Load image
            with Image.open(os.path.join(img_path, img_file)) as img:
                img = img.resize((512, 512))
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
            
            # Load prompt from JSON
            with open(os.path.join(img_path, json_file), 'r') as f:
                data = json.load(f)
                if 'prompt_short' in data:
                    prompt = data['prompt_short']
                else:
                    prompt = data['prompt']
                batch_prompts.append(prompt)
        
        gen_images_array = np.stack(batch_images)
        images_int = (gen_images_array * 255).astype("uint8")
        images_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2)
        
        clip_score_val = clip_score_fn(images_tensor, batch_prompts).detach().cpu().numpy()
        total_score += clip_score_val.sum()
        count += images_tensor.shape[0]
    
    avg_clip_score = total_score / count
    print(f"Average CLIP score: {avg_clip_score}")
    return avg_clip_score

# Function to calculate SSIM in batches
def calculate_ssim_in_batches(img_path, batch_size=100):
    total_ssim = 0
    count = 0
    
    for gen_images_array, ref_images_array in load_images_in_batches(img_path, batch_size):
        for gen_img, ref_img in zip(gen_images_array, ref_images_array):
            gen_img = (gen_img * 255).astype(np.uint8)
            ref_img = (ref_img * 255).astype(np.uint8)
            ssim_value = ssim(gen_img, ref_img, multichannel=True, win_size=7, channel_axis=-1)
            total_ssim += ssim_value
            count += 1
    
    avg_ssim = total_ssim / count
    print(f"Average SSIM: {avg_ssim}")
    return avg_ssim

# Function to calculate PSNR in batches
def calculate_psnr_in_batches(img_path, batch_size=100):
    total_psnr = 0
    count = 0
    
    for gen_images_array, ref_images_array in load_images_in_batches(img_path, batch_size):
        for gen_img, ref_img in zip(gen_images_array, ref_images_array):
            gen_img = (gen_img * 255).astype(np.uint8)
            ref_img = (ref_img * 255).astype(np.uint8)
            psnr_value = psnr(gen_img, ref_img)
            total_psnr += psnr_value
            count += 1
    
    avg_psnr = total_psnr / count
    print(f"Average PSNR: {avg_psnr}")
    return avg_psnr

# Function to calculate LPIPS in batches
def calculate_lpips_in_batches(img_path, batch_size=100):
    lpips_fn = lpips.LPIPS(net='alex').to('cuda:0')
    total_lpips = 0
    count = 0
    
    for gen_images_array, ref_images_array in load_images_in_batches(img_path, batch_size):
        for gen_img, ref_img in zip(gen_images_array, ref_images_array):
            gen_img_tensor = torch.from_numpy(gen_img).permute(2, 0, 1).unsqueeze(0).cuda().float() * 2 - 1
            ref_img_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0).cuda().float() * 2 - 1
            lpips_value = lpips_fn(gen_img_tensor, ref_img_tensor)
            total_lpips += lpips_value.item()
            count += 1
    
    avg_lpips = total_lpips / count
    print(f"Average LPIPS: {avg_lpips}")
    return avg_lpips

# Function to calculate FID in batches
def calculate_fid_in_batches(img_path, batch_size=100):
    fid = FrechetInceptionDistance(feature=2048)
    
    for gen_images_array, ref_images_array in tqdm(load_images_in_batches(img_path, batch_size), desc='batch'):
        gen_images_tensor = torch.tensor((gen_images_array * 255).astype(np.uint8)).permute(0, 3, 1, 2)
        ref_images_tensor = torch.tensor((ref_images_array * 255).astype(np.uint8)).permute(0, 3, 1, 2)
        fid.update(ref_images_tensor, real=True)
        fid.update(gen_images_tensor, real=False)
    
    fid_score = fid.compute().item()
    print(f"Average FID score: {fid_score}")
    return fid_score

# Function to calculate Inception Score in batches
def calculate_inception_score_in_batches(img_path, batch_size=100):
    inception_score = InceptionScore()
    
    for gen_images_array, _ in tqdm(load_images_in_batches(img_path, batch_size)):
        gen_images_tensor = torch.tensor((gen_images_array * 255).astype(np.uint8)).permute(0, 3, 1, 2)
        inception_score.update(gen_images_tensor)
    
    is_mean, is_std = inception_score.compute()
    print(f"Inception Score: {is_mean.item()} ± {is_std.item()}")
    return is_mean.item(), is_std.item()

# Main function to calculate all metrics in batches
def calculate_all_metrics(root, batch_size=100):
    img_path = root
    metrics_path = os.path.join(root, f'metrics.json') 
    dict  = {}
    print(img_path)
    
    dict['clip_score'] = float(calculate_clip_in_batches(img_path, batch_size))
    dict['ssim'] = float(calculate_ssim_in_batches(img_path, batch_size))
    dict['psnr'] = float(calculate_psnr_in_batches(img_path, batch_size))
    dict['lpips'] = float(calculate_lpips_in_batches(img_path, batch_size))
    dict['fid'] = float(calculate_fid_in_batches(img_path, batch_size))
    is_mean, is_std = calculate_inception_score_in_batches(img_path, batch_size)
    dict['inception_score'] = {
        "mean": float(is_mean),
        "std": float(is_std)
    }

    print("----------------------------------------------")

    json_str = json.dumps(dict, indent=4)
    json_line = json_str.splitlines()
    print("metric path", metrics_path) 

    # Write each line to the file
    with open(metrics_path, 'w') as json_file:
        for line in json_line:
            json_file.write(line + '\n')

if __name__ == "__main__":
    root = '/home/ecomapper/data/datasets/logs/' 
    model_names = [
        "Final_diffusionsat_weather_2_epochs",

    ]
    for name in model_names:
        image_folder = os.path.join(root, name, '2', '512')
        calculate_all_metrics(image_folder, batch_size = 100)
