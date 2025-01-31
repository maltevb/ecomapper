from datetime import datetime
import pandas as pd
import torch
import calendar
import random
import os


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


def convert_cloud_info(cloud_cover):
        if cloud_cover > 0.7:
            return "very cloudy "
        elif 0.5 < cloud_cover <= 0.7:
            return "cloudy "
        elif 0.3 < cloud_cover <= 0.5:
            return "partially cloudy "
        else:
            return ""

###

# diffsat, diffsat weather
# sd3, sd3 weather 

# sd3 numerical  remove rounding

# sd3 masking .

# a staeliite croplandn jan usa, temp -20
# a staeliite croplandn ja} usa, {temp +100}




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


def normalize_metadata(
    metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=10, min_temp=-60, max_temp=60, max_prep = 120, max_rad=40,scale=1000
):
    (
        lon,
        lat,
        gsd,
        cloud_cover,
        year,
        month,
        day,
        avg_temp,
        avg_prep,
        avg_rad
    ) = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale
    gsd = gsd / max_gsd * scale
    cloud_cover = cloud_cover * scale
    year = year / (2100 - base_year) * scale
    month = month / 12 * scale
    day = day / 31 * scale

    avg_temp = ((avg_temp - min_temp)/(max_temp - min_temp)) * (scale)
    avg_prep = (avg_prep/max_prep) * (scale)
    avg_rad = (avg_rad/max_rad) * (scale)
    return torch.tensor(
        [
            lon,
            lat,
            gsd,
            cloud_cover,
            year,
            month,
            day,
            avg_temp,
            avg_prep,
            avg_rad
        ]
    )
def get_metadata(img_data):
    cloud_coverage = img_data["cloud_coverage"]

    timestamp = img_data["date"]
    if len(timestamp.split("-")) == 2:
        timestamp = f"{timestamp}-01T00:00:00Z"
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    year = dt.year
    month = dt.month
    day = dt.day
    average_temp = img_data['average_temp']
    average_perp = img_data['average_prep']
    average_rad = img_data['average_rad']
    latitude = img_data["latitude"]
    longitude = img_data["longitude"]
    gsd = 20

    metadata = [longitude, latitude, gsd, cloud_coverage, year, month, day,average_temp,average_perp,average_rad]
    return normalize_metadata(metadata)


def convert_all_data(img_data, use_numerical_values = True):
    land_type = img_data["type"]
    month, year = convert_date_string(img_data["date"])
    cloud_info = convert_cloud_info(img_data["cloud_coverage"])
    state, country = convert_location(img_data['location_address'])
    
    temp = get_descriptive_word(img_data['average_temp'], 'average_temp') 
    prep = get_descriptive_word(img_data['average_prep'], 'average_prep')
    rad = get_descriptive_word(img_data['average_rad'], 'average_rad')

    if use_numerical_values:
        temp = round(img_data['average_temp'])     
        prep = round(img_data['average_prep']) 
        rad = round(img_data['average_rad'])

    return [land_type, month, year, cloud_info, state, country, temp, prep, rad]
 


class CaptionBuilder():

    def __init__(self, use_numerical_values = False, masks = None):
        self.use_numerical_values = use_numerical_values
        self.masks = masks
        if self.masks == None:
            self.initialize_masks() 

    def initialize_masks(self):
        self.masks = {
            'land' : [0.0, 1.0], #[skip, land_type]
            'cloud': [0.0, 1.0], #[skip, cloud_info]
            'date': [0.0, 1.0, 0.0, 0.0], #[skip, month and year, month, year]
            'location': [0.0, 1.0, 0.0], # [skip, state and country ,  country]
            'climate' : [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #[skip, (tmp, prp, rad), (tmp, prp), (tmp, rad),(tmp), (prp, rad), (prp), (rad)]
        }


    def generate(self, img_data):

        (
            land_type, month, year, 
            cloud_info, state, country,
            temp, prep, rad 
        ) = convert_all_data(img_data, self.use_numerical_values)
        
        metadata = get_metadata(img_data)

        skip = ""
        land_options = [skip, f" of {land_type}"]
        cloud_options = [skip , cloud_info]
        date_options = [skip, f" on {month} {year}", f" on {month}", f" on {year}"]
        

        if state == "":
            location_options = [skip, f" in {country}", f" in {country}"]
        else:
            location_options = [skip, f" in {state}, {country}", f" in {country}"]

        if self.use_numerical_values:
            climate_caption_options = [
                skip, 
                f" The average temperature over the last month was {temp} C. with an average precipitation {prep} mm, and an average daily solar radiation of {rad} W/m2.",
                f" The average temperature over the last month was {temp} C. with an average precipitation {prep} mm",
                f" The average temperature over the last month was {temp} C. and an average daily solar radiation of {rad} W/m2.",
                f" The average temperature over the last month was {temp} C.",
                f" The average percipitation over the last month was {prep} mm, and an average daily solar radiation of {rad} W/m2.", 
                f" The average percipitation over the last month was {prep} mm.", 
                f" The average daily solar radiation over the last month was {rad} W/m2."
            ]
        else:
            climate_caption_options = [
                    skip,
                    f" The average temperature over the last month was {temp}, with {prep} precipitation, and {rad} daily solar radiation.",
                    f" The average temperature over the last month was {temp}, with {prep} precipitation",
                    f" The average temperature over the last month was {temp}, and {rad} daily solar radiation.",
                    f" The average temperature over the last month was {temp}." ,
                    f" The average percipitation over the last month was {prep}, and {rad} daily solar radiation.",
                    f" The average percipitation over the last month was {prep}.",
                    f" The average daily solar radiation over the last month was {rad}."
            ]


        selected_cloud_info = random.choices(cloud_options, weights=self.masks['cloud'])[0]
        selected_land_info = random.choices(land_options, weights=self.masks['land'])[0]
        selected_date_info = random.choices(date_options, weights=self.masks['date'])[0]
        selected_location_info = random.choices(location_options, weights=self.masks['location'])[0]

        selected_climate_info = random.choices(climate_caption_options, weights=self.masks['climate'])[0]

        base_caption = f"a {selected_cloud_info}satellite image{selected_land_info}{selected_location_info}{selected_date_info}."
        climate_caption = selected_climate_info


        return base_caption, base_caption + climate_caption, metadata



            











        



        








