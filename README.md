# DATA H195A - Data Science Thesis Project - Using Deep Learning for Climate Forecasting

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexander-tavasoli/Data-Science-Thesis.git/master)

_Omeed Tavasoli_

This honors thesis will explore the intersection of climate science and deep learning with most of the focus on improving seasonal forecasting. 

## Directory Structure

This repo contains a few subfolders which contain the elements of this thesis.

| Folder | Description |
|-----|-----|
| `data`  | contains all reanalysis and forecast data which should be in netCDF4 format  |
| `notebooks`  | contains jupyter notebooks for analysis of data as well as pre-processing scripts |
| `output`  | output folder will contain images generated in data analysis notebooks  |

## Environment and packages
The code was tested and written in Python 3.7.3 on Ubuntu. It makes use of the following Python 3.7.3 packages (these can be found in `requirements.txt` or `environment.yml` - see below for details about running in a conda environment):
+ **numpy**: 1.16.4
+ **pandas**: 0.24.2
+ **matplotlib**: 3.1.0
+ **scipy**: 1.2.1
+ **basemap**: 1.2.0
+ **xarray**: 0.12.3
+ **xesmf**: 4.32.2
+ **datetime**: 8.1
+ **python-cdo**: 1.5.3


To create a conda virtual environment with the above dependencies, you can use the `environment.yml` file. Navigate to the base directory and type the command:

```bash
$ conda env create -n <your-environment-name-here> -f environment.yml

```
to create an environment with the name of your choosing, containing all of the dependencies required for the notebooks.

## Environment Variables

Be sure to have these environment variables instantiated in your .bashrc or .bashprofile files where your_root is the root before the repository

-- NCEP_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Reanalysis/NCEP_Reanalysis_2/'

-- BERKELEY_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Reanalysis/Berkeley_Earth/'

-- CFS_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Forecasts/CFS/'

-- NMME_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Forecasts/NMME/'

-- SEAS_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Forecasts/SEAS/'

-- GRID_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Grid_Data/'

-- ERA_DIR= your_root + '/CFSv2-and-SEAS5-Analysis/data/Reanalysis/ERA-5/'

## Data

Given the large nature of climate data sets, I cannot store the files on github. As a future task, I will need to write scripts that will download the necessary data automatically for the user.

## Notebooks
The notebooks included thus far only conduct a basic analysis of the climate datasets I am interested in using. More will be added that will actually delve into the deep learning aspects of my project.

## Sources

Chris Pyles for structure of README thus far.
