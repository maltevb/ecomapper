# Ecomapper Dataset Card

## Dataset Summary
The EcoMapper dataset comprises over 2.9 million satellite images accompanied by climate metadata. It includes both RGB imagery and selected multispectral channels (B6 – Red Edge 2, B8 – NIR, B11 – SWIR1) sourced from the Copernicus Sentinel satellite missions. The dataset spans diverse land cover types and multiple temporal points. The training set features 98,930 unique locations, each with 24 months of data, while the test set includes 5,494 locations covering 96 months. Each timestamp is paired with weather-related metadata, including temperature, solar radiation, and precipitation. The imagery is derived from the Sentinel-2 mission, supported by the **European Space Agency’s** (ESA) Network of Resources (NoR) program.
<img width="1733" alt="Dataset (1)" src="https://github.com/user-attachments/assets/6b5a2528-2bfb-41f5-a230-73aace1eb6e0" />

### Dataset Version and Maintenance
#### Maintenance Status
- **Regularly Updated:** Yes
- **Last Updated:** 05/2025

## Supported Tasks
- **Agricultural Monitoring**: Analyzing land cover changes, identifying crop types, and monitoring vegetation growth.
- **Environmental Monitoring**: Studying land surface dynamics, vegetation, and ecosystem changes.
- **Remote Sensing**: Various remote sensing applications related to environmental research.

## Sensitivity of Data
- None
- 
## Authorship

### Publishers

#### Industry Type(s)
- **Type:** Not-for-profit - Tech

#### Contact Detail(s)
- 
## Dataset Structure

- **Location**: Latitude and longitude of each sampled location.
- **Time Series**: 24 months of data, one entry for each month.
- **Land Cover Type**: Classifications based on land cover at the sampled location.
- **Cloud Coverage (%)**: Monthly cloud coverage for each data point.
- **Weather Data**: Includes solar radiation, precipitation, and temperature for each location over the 24-month period.
- **Sentinel Data**: Spectral bands including Red, Green, Blue bands derived from Sentinel satellite imagery.

### Dataset Snapshot
| Category             | Data                                       |
|----------------------|--------------------------------------------|
| Size of Training Dataset      | 690 GB                                    |
| Size of Test Dataset   | 133 GB        | 
| Number of Instances in Train Dataset  | 99,000 locations × 24 months = 2,376,000  |
| Number of Instances in Test Dataset  | 5,500 locations × 96 months = 528,000  |
| Labeled Classes      | 15  |

### Directory Structure
The dataset is organized into three batches, each spanning two years and containing 28,000 locations. For each location, there are 48 files: 24 image files and 24 JSON metadata files.

```
Dataset/
├── Train/
│   ├── 2017_2018/
|   |   ├── Batch-1/
│   │   |   ├── Location_1/
│   │   |   │   ├── 1_2017_01_01.png
│   │   |   │   ├── 1_2017_01_01.json
│   │   |   │   ├── 1_2017_02_01.png
│   │   |   │   ├── 1_2017_02_01.json
│   │   |   │   └── ... (44 more files)
│   │   |   ├── Location_2/
│   │   |   │   └── ...
│   │   |   └── ... (27,998 more locations)
|   |   ├── Batch-1-ms/
│   │   |   ├── Location_1/
│   │   |   │   ├── 1_2017_01_01.png
│   │   |   │   ├── 1_2017_02_01.png
│   │   |   │   └── ... (22 more files)
│   │   |   ├── Location_2/
│   │   |   │   └── ...
│   │   |   └── ... (27,998 more locations)
|   |   |   
│   |   ├── Batch_2/
│   │   └── ...
│   |   ├── Batch_2-ms/
│   │   └── ...
│   |   ├── Batch_3/
│   │   └── ...
│   |   ├── Batch_3-ms/
│   │   └── ...
|   ├── 2019_2020/
│   |   ├── Batch_4/
│   │   └── ...
│   |   ├── Batch_5/
│   │   └── ...
|   |
└── Test/
    ├── Location_1/
    |    ├── 1_2017_01_01.png
    |    ├── 1_2017_01_01.json
    |    ├── 1_2017_02_01.png
    |    ├── 1_2017_02_01.json
    |    └── ... (140 more files)
    ├── Location_2/
    │   └── ...
    └── ... (4,998 more locations)
```

#### List of classes:
- Evergreen Needleleaf Forest
- Evergreen Broadleaf Forest
- Deciduous Needleleaf Forest
- Deciduous Broadleaf Forest
- Mixed Forest
- Woodland
- Wooded Grassland
- Grassland
- Cropland
- Urban and Built-up
- Cropland/Natural Vegetation Mosaic
- Savanna
- Shrubland
- Grassland/Shrubland Mosaic
- Barren or Sparsely Vegetated
### Example Entry
| Latitude  | Longitude | Month-Year | Land Cover Type | Cloud Coverage (%) | Weather Data (Solar Radiation, Precipitation, Temp)    | Sentinel Band Data |
|-----------|-----------|------------|-----------------|--------------------|-------------------------------------------------------|---------------------|
| 34.0522   | -118.2437 | 2022-01    | Urban           | 15                 | Solar Rad: 5.5 kWh/m², Precip: 30 mm, Temp: 20°C       | [2, 3, 4], [6,8,11]|

### Data Fields
| Field Name         | Field Value         | Description                       |
|--------------------|---------------------|-----------------------------------|
| Latitude           | 34.0522            | Geographical latitude            |
| Longitude          | -118.2437          | Geographical longitude           |
| Month-Year         | 2022-01            | Temporal data                    |
| Land Cover Type    | Urban              | Land cover classification        |
| Cloud Coverage (%) | 15                 | Percentage of cloud coverage     |
| Weather Data       | 5.5 kWh/m², 30 mm, 20°C | Solar radiation, precipitation, temperature |
| Sentinel Band Data | [2, 3, 4, 6, 8, 11]| Multispectral satellite bands    |
## Languages
- English (metadata)


## License
- [http://creativecommons.org/licenses/by/4.0]

## Citation
If you use this dataset, please cite the following:

## Acknowledgements
This dataset was developed with the support of the **European Space Agency (ESA)**. Special thanks to **Sentinel Hub** for providing the satellite data.

## Funding
This work was supported by the **European Space Agency (ESA)** through their sponsorship of the **Sentinel Hub** initiative.

## Contributions
If you'd like to contribute to the dataset or provide feedback, please contact .

## Usage
- The dataset can be accessed via [**].
