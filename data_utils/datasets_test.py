from create_captions import CaptionBuilder

from dateutil.relativedelta import relativedelta

from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
import json
import os



def check_date_difference_within_limit(image_path1, image_path2, month_limit=6):
    # Extract the date part of the filenames
    date_str1 = image_path1.split('/')[-1].split('_')[1].replace('.png', '')
    date_str2 = image_path2.split('/')[-1].split('_')[1].replace('.png', '')
   
    date1 = datetime.strptime(date_str1, '%Y-%m')
    date2 = datetime.strptime(date_str2, '%Y-%m')
    
    # Calculate the difference between the two dates
    difference = relativedelta(date2, date1)
    
    # Check if the difference is within the specified month limit
    if abs(difference.years * 12 + difference.months) <= month_limit:
        return True
    return False

def is_valid_row(row):
    keys_to_check = ["date", "latitude", "longitude", "location_address",
        "cloud_coverage", "average_temp", "average_prep", "average_rad"
    ]
    if not os.path.exists(row['img_path']):
        
        return 1
    for key in keys_to_check:
        value = row[key]
        if pd.isna(value):
            return 2

    return 3

def build_transform(normalize, img_size):

    t = []

    t.append(transforms.ToTensor())
    t.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))

    if normalize:
        t.append(transforms.Normalize([0.5], [0.5]))
            
    return transforms.Compose(t)

def process_csv(csv_path, dataset_name, root, is_train):
    csv_data = pd.read_csv(csv_path)
    data = []
    missing = 0
    nan_row = 0
    for _, row in csv_data.iterrows():
        row_data = row.to_dict()
        row_data['img_path'] = os.path.join(root, dataset_name, str(row_data['location_id']), row_data['img_path'])
        k = is_valid_row(row_data)
        if k==1:
            missing += 1
            continue
        if k ==2:
            nan_row +=1
            continue

        data.append(row_data)
    print(f" for {dataset_name} there are {missing} missing file!")
    print(f" for {dataset_name} there are {nan_row} Nan file!")
    return data


# Helper function to process each CSV file
def process_csv_sequence(csv_path, dataset_name, root, past_month_max):
    csv_data = pd.read_csv(csv_path)
    data = []
    past_and_current_index = []
    idx = 0
    
    for _, row in csv_data.iterrows():
        row_data = row.to_dict()
        row_data['img_path'] = os.path.join(root, dataset_name, str(row_data['location_id']), row_data['img_path'])
        
        if not is_valid_row(row_data):
            continue
        
        data.append(row_data)
        
        past_img_candidates_idx = []
        for i in range(1, past_month_max + 1):
            past_img_idx = idx - i
            if past_img_idx < 0:  # Ensure we're not using indices from previous CSVs
                break
            if data[idx]['location_id'] != data[past_img_idx]['location_id']:
                break
            if not check_date_difference_within_limit(data[idx]['img_path'], data[past_img_idx]['img_path'], month_limit=past_month_max):
                break

            past_img_candidates_idx.append(past_img_idx)

        if len(past_img_candidates_idx) > 0:
            past_and_current_index.append((past_img_candidates_idx, idx))

            

        idx += 1
    
    return data, past_and_current_index, idx


class Satellite_Dataset(Dataset):
    def __init__(
        self,
        csv_paths: list,
        dataset_names: list,
        root: str,
        use_numerical_values = True,
        masks = None,
        img_size = 512,
        is_train = True,
    ):
        super().__init__()

        self.data = []

        # Parallelize the CSV reading and processing
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_csv, csv_paths, dataset_names, [root]*len(csv_paths), [is_train]*len(csv_paths)), total=len(csv_paths)))
        # Combine all the processed data from different files
        for result in results:
            self.data.extend(result)
        self.caption_builder = CaptionBuilder(use_numerical_values, masks) 
        self.transforms = build_transform(normalize=is_train, img_size=img_size)

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data['img_path']
        pil_img = Image.open(img_path)
        img_as_tensor = self.transforms(pil_img)

        caption_clip, caption_t5, metadata = self.caption_builder.generate(img_data)

        return {
            "img": img_as_tensor,
            "caption_clip": caption_clip,
            "caption_t5": caption_t5,
            "metadata" : metadata,
            "img_path" : img_path
        }

    def __len__(self):
        return len(self.data)



class Sequence_Satellite_Dataset(Dataset):
    def __init__(
        self,
        csv_paths: list,
        dataset_names: list,
        root: str,
        past_month_max = 6,
        use_numerical_values = True,
        masks = None,
        img_size = 512,
        is_train = True,
    ):
        super().__init__()

        self.data = []
        self.past_and_current_index = []
        current_offset = 0  # Start offset for the first file

        # Parallelize the CSV processing and data population using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_csv_sequence, csv_paths, dataset_names, [root]*len(csv_paths), [past_month_max]*len(csv_paths)), total=len(csv_paths)))

        # Combine results from different CSV files
        for result in tqdm(results):
            data, past_and_current_index, new_offset = result
            self.data.extend(data)

            # Adjust past_and_current_index after combining results
            for i, (past_indices, current_index) in enumerate(past_and_current_index):
                adjusted_past_indices = [past_idx + current_offset for past_idx in past_indices]
                self.past_and_current_index.append((adjusted_past_indices, current_index + current_offset))

            current_offset += new_offset  # Update the offset for the next file
        
        self.caption_builder = CaptionBuilder(use_numerical_values, masks) 
        self.transforms = build_transform(normalize=is_train, img_size=img_size)
        self.transforms_control_img = build_transform(normalize=False, img_size=img_size)
        

    def __getitem__(self, index):
        past_img_candidates_idx, current_idx = self.past_and_current_index[index]

        past_idx = random.choice(past_img_candidates_idx) 

        img_data = self.data[current_idx]
        past_img_data = self.data[past_idx]

        pil_img = Image.open(img_data['img_path'])
        img_as_tensor = self.transforms(pil_img)

        pil_past_img = Image.open(past_img_data['img_path'])
        past_img_as_tensor = self.transforms_control_img(pil_past_img)

        caption_clip, caption_t5, _= self.caption_builder.generate(img_data)

        caption_clip_controlnet, caption_t5_controlnet, _ = self.caption_builder.generate(past_img_data)

        return {
            "img": img_as_tensor,
            "past_img" : past_img_as_tensor,
            "caption_clip": caption_clip,
            "caption_t5": caption_t5,
            "control_caption_clip" : caption_clip_controlnet,
            "control_caption_t5": caption_t5_controlnet, 
        
        }

    def __len__(self):
        return len(self.past_and_current_index)


if __name__ == "__main__":

    #### EXAMPLE
    csv_paths = [
        "/home/ecomapper/data/datasets/seed_14_final.csv", 
    "/home/ecomapper/data/datasets/seed_28_final.csv", 
    "/home/ecomapper/data/datasets/seed_42_final.csv",
     "/home/ecomapper/data/datasets/seed_56_final.csv",
    "/home/ecomapper/data/datasets/seed_84_final.csv",
      "/home/ecomapper/data/datasets/seed_98_final.csv",
    "/home/ecomapper/data/datasets/seed_112_final.csv",
      "/home/ecomapper/data/datasets/seed_126_final.csv",
    "/home/ecomapper/data/datasets/seed_140_final.csv",
      "/home/ecomapper/data/datasets/seed_70_train.csv" 
    
    
    ]
    dataset_names = [
        "proportional_sampled_points_seed_14", 
        "proportional_sampled_points_seed_28", 
        "proportional_sampled_points_seed_42",
     "proportional_sampled_points_seed_56", 
     "proportional_sampled_points_seed_84",
       "proportional_sampled_points_seed_98",
     "proportional_sampled_points_seed_112",
       "proportional_sampled_points_seed_126",
         "proportional_sampled_points_seed_140",
           "proportional_sampled_points_seed_70_new"]


    
    csv_paths = ["/home/ecomapper/data/datasets/seed_14_final.csv",
                  "/home/ecomapper/data/datasets/seed_70_train.csv" ,
                  "/home/ecomapper/data/datasets/seed_126_final.csv",
                  "/home/ecomapper/data/datasets/seed_28_final.csv", 
                    "/home/ecomapper/data/datasets/seed_42_final.csv",
                    "/home/ecomapper/data/datasets/seed_56_final.csv",
                    "/home/ecomapper/data/datasets/seed_84_final.csv",
                    "/home/ecomapper/data/datasets/seed_112_final.csv",
                    "/home/ecomapper/data/datasets/seed_140_final.csv",


                  ]
    dataset_names = ["proportional_sampled_points_seed_14",
                      "proportional_sampled_points_seed_70_new", 
                      "proportional_sampled_points_seed_126",
                      "proportional_sampled_points_seed_28", 
                        "proportional_sampled_points_seed_42",
                        "proportional_sampled_points_seed_56", 
                        "proportional_sampled_points_seed_84",
                        "proportional_sampled_points_seed_112",
                        "proportional_sampled_points_seed_140",

                      ]

    root = "/home/ecomapper/data/datasets"
    dataset = Satellite_Dataset(csv_paths, dataset_names, root)

    train_loader = DataLoader(dataset, batch_size=32, num_workers = 32, pin_memory = True)
    a = 0
    for batch in tqdm(train_loader):
        b = batch['img']
        a = a + b.shape[0]

    print("final lenght :", a)

        
        