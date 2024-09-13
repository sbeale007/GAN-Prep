# Prepare climatology data for input into GAN 
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def standardize(ds: xr.Dataset):
    # standardization is done based on mean and standard deviation of training data
    # testing data is normalized using that mean and std
    tens = torch.from_numpy(np.array(ds.prec))
    
    if tens.shape
    
    worldclim_tens = worldclim_tens.where(np.isnan(prism_coarse_tens)==False, np.nan)
    wrf_tens = wrf_tens.where(np.isnan(prism_coarse_tens)==False, np.nan)
    dem_tens = dem_tens.where(np.isnan(prism_tens)==False, np.nan)
    lat_tens = lat_tens.where(np.isnan(prism_tens)==False, np.nan)
    lon_tens = lon_tens.where(np.isnan(prism_tens)==False, np.nan)
    coast_tens = coast_tens.where(np.isnan(prism_tens)==False, np.nan)
    
    prism_prec = prism['prec']
    worldclim_prec = worldclim['prec']
    
    worldclim_prec = worldclim['prec']
    coast_dist = coast['dist']
    dem_dem = dem['dem']
    lat_lat = lat['lat']
    lon_lon = lon['lon']
    
    prism_mean = np.nanmean(prism_tens)
    worldclim_mean = np.nanmean(worldclim_tens)
    coast_mean = np.nanmean(coast_tens)
    dem_mean = np.nanmean(dem_tens)
    lat_mean = np.nanmean(lat_tens)
    lon_mean = np.nanmean(lon_tens)
    
    # calculating standard deviation 

    prism_std = np.nanstd(prism_tens)
    worldclim_std = np.nanstd(worldclim_tens)
    coast_std = np.nanstd(coast_tens)
    dem_std = np.nanstd(dem_tens)
    lat_std = np.nanstd(lat_tens)
    lon_std = np.nanstd(lon_tens)
    
    prism['prec'] = (prism_prec - prism_mean)/prism_std
worldclim['prec'] = (worldclim_prec - worldclim_mean)/worldclim_std
coast['dist'] = (coast_dist - coast_mean)/coast_std
dem['dem'] = (dem_dem - dem_mean)/dem_std
lat['lat'] = (lat_lat - lat_mean)/lat_std
lon['lon'] = (lon_lon - lon_mean)/lon_std

# standardizing test variables 
prism_prec = prism_test['prec']
worldclim_prec = worldclim_test['prec']
coast_dist = coast_test['dist']
dem_dem = dem_test['dem']
lat_lat = lat_test['lat']
lon_lon = lon_test['lon']

prism_test['prec'] = (prism_prec - prism_mean)/prism_std
worldclim_test['prec'] = (worldclim_prec - worldclim_mean)/worldclim_std
coast_test['dist'] = (coast_dist - coast_mean)/coast_std
dem_test['dem'] = (dem_dem - dem_mean)/dem_std
lat_test['lat'] = (lat_lat - lat_mean)/lat_std
lon_test['lon'] = (lon_lon - lon_mean)/lon_std
    return 

def make_tiles():
    prism_pt = np.array(prism.prec)
worldclim_pt = np.array(worldclim.prec)
coast_pt = np.array(coast.dist)
dem_pt = np.array(dem.dem)
lat_pt = np.array(lat.lat)
lon_pt = np.array(lon.lon)

prism_test_pt = np.array(prism_test.prec)
worldclim_test_pt = np.array(worldclim_test.prec)
coast_test_pt = np.array(coast_test.dist)
dem_test_pt = np.array(dem_test.dem)
lat_test_pt = np.array(lat_test.lat)
lon_test_pt = np.array(lon_test.lon)

x_fine = np.arange(0,prism_pt.shape[0]-128, 8)
y_fine = np.arange(0,prism_pt.shape[1]-128, 8)

x_coarse = np.arange(0,worldclim_pt.shape[0]-32, 2)
y_coarse = np.arange(0,worldclim_pt.shape[1]-32, 2)

x_fine_test= np.arange(0,prism_test_pt.shape[0]-128, 8)
y_fine_test = np.arange(0,prism_test_pt.shape[1]-128, 8)

x_coarse_test = np.arange(0,worldclim_test_pt.shape[0]-32, 2)
y_coarse_test = np.arange(0,worldclim_test_pt.shape[1]-32, 2)

tiles_fine_prism = []
tiles_fine_coast = []
tiles_fine_dem = []
tiles_fine_lat = []
tiles_fine_lon = []
for i in range(0, len(y_fine)):
    prism = prism_pt[:,0+y_fine[i]:128+y_fine[i]]
    coast = coast_pt[:,0+y_fine[i]:128+y_fine[i]]
    dem = dem_pt[:,0+y_fine[i]:128+y_fine[i]]
    lat = lat_pt[:,0+y_fine[i]:128+y_fine[i]]
    lon = lon_pt[:,0+y_fine[i]:128+y_fine[i]]
    for j in range(0, len(x_fine)):
        tiles_fine_prism.append(prism[0+x_fine[j]:128+x_fine[j]])
        tiles_fine_coast.append(coast[0+x_fine[j]:128+x_fine[j]])
        tiles_fine_dem.append(dem[0+x_fine[j]:128+x_fine[j]])
        tiles_fine_lat.append(lat[0+x_fine[j]:128+x_fine[j]])
        tiles_fine_lon.append(lon[0+x_fine[j]:128+x_fine[j]])
        
        tiles_fine_prism_test = []
tiles_fine_coast_test = []
tiles_fine_dem_test = []
tiles_fine_lat_test = []
tiles_fine_lon_test = []
for i in range(0, len(y_fine_test)):
    prism = prism_test_pt[:,0+y_fine_test[i]:128+y_fine_test[i]]
    coast = coast_test_pt[:,0+y_fine_test[i]:128+y_fine_test[i]]
    dem = dem_test_pt[:,0+y_fine_test[i]:128+y_fine_test[i]]
    lat = lat_test_pt[:,0+y_fine_test[i]:128+y_fine_test[i]]
    lon = lon_test_pt[:,0+y_fine_test[i]:128+y_fine_test[i]]
    for j in range(0, len(x_fine_test)):
        tiles_fine_prism_test.append(prism[0+x_fine_test[j]:128+x_fine_test[j]])
        tiles_fine_coast_test.append(coast[0+x_fine_test[j]:128+x_fine_test[j]])
        tiles_fine_dem_test.append(dem[0+x_fine_test[j]:128+x_fine_test[j]])
        tiles_fine_lat_test.append(lat[0+x_fine_test[j]:128+x_fine_test[j]])
        tiles_fine_lon_test.append(lon[0+x_fine_test[j]:128+x_fine_test[j]])
        
tiles_coarse_worldclim = []
for i in range(0, len(y_coarse)):
    worldclim = worldclim_pt[:,0+y_coarse[i]:32+y_coarse[i]]
    for j in range(0, len(x_fine)):
        tiles_coarse_worldclim.append(worldclim[0+x_coarse[j]:32+x_coarse[j]])
        
        tiles_coarse_worldclim_test = []
for i in range(0, len(y_coarse_test)):
    worldclim = worldclim_test_pt[:,0+y_coarse_test[i]:32+y_coarse_test[i]]
    for j in range(0, len(x_fine_test)):
        tiles_coarse_worldclim_test.append(worldclim[0+x_coarse_test[j]:32+x_coarse_test[j]])
    return 

def remove_nan():
    good_tiles_prism = []
good_tiles_dem = []
good_tiles_coast = []
good_tiles_lat = []
good_tiles_lon = []

good_tiles_worldclim = []

for i in range(0, len(tiles_fine_prism)):
    ind = np.argwhere(np.isnan(tiles_fine_prism[i]))
    percent_nan_prism = ind.shape[0]/size_fine *100
    ind = np.argwhere(np.isnan(tiles_coarse_worldclim[i]))
    percent_nan_worldclim = ind.shape[0]/size_coarse *100
    if ((percent_nan_prism == 0) & (percent_nan_worldclim==0)):
        good_tiles_prism.append(tiles_fine_prism[i])
        good_tiles_dem.append(tiles_fine_dem[i])
        good_tiles_coast.append(tiles_fine_coast[i])
        good_tiles_lat.append(tiles_fine_lat[i])
        good_tiles_lon.append(tiles_fine_lon[i])
        
        good_tiles_worldclim.append(tiles_coarse_worldclim[i])
        
        good_tiles_prism_test = []
good_tiles_dem_test = []
good_tiles_coast_test = []
good_tiles_lat_test = []
good_tiles_lon_test = []

good_tiles_worldclim_test = []

for i in range(0, len(tiles_fine_prism_test)):
    ind = np.argwhere(np.isnan(tiles_fine_prism_test[i]))
    percent_nan_prism = ind.shape[0]/size_fine *100
    ind = np.argwhere(np.isnan(tiles_coarse_worldclim_test[i]))
    percent_nan_worldclim = ind.shape[0]/size_coarse *100
    if ((percent_nan_prism == 0) & (percent_nan_worldclim==0)):
        good_tiles_prism_test.append(tiles_fine_prism_test[i])
        good_tiles_dem_test.append(tiles_fine_dem_test[i])
        good_tiles_coast_test.append(tiles_fine_coast_test[i])
        good_tiles_lat_test.append(tiles_fine_lat_test[i])
        good_tiles_lon_test.append(tiles_fine_lon_test[i])
        
        good_tiles_worldclim_test.append(tiles_coarse_worldclim_test[i])
    return 


# low resolution variable and covariates 
lr = ['WRF', 'WorldClim']

hr = ['PRISM']

# high resolution covariates
hr_cov = ['DEM', 'lat', 'lon', 'coast']

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

var = 'tmax'

# directory to get data from 
dir = 'C:/Users/SBEALE/Desktop/Cropped_Coarsened_WRF_PRISM/no_overlap/'

# directory to save files to 
save_dir = 'C:/Users/SBEALE/Desktop/test/'

# loop through all months and create pytorch files for training 
for i in months:
    # open all relevant datasets (prism, worldclim, wrf, hr_cov) 
    prism = xr.open_dataset(dir + var + '_' + months[i] + '_' + hr[0] + '.nc')
    wrf = xr.open_dataset(dir + var + '_' + months[i] + '_' + lr[0] + '.nc')
    worldclim = xr.open_dataset(dir + var + '_' + months[i] + '_' + lr[1] + '.nc') 
    
    dem = xr.open_dataset(dir + hr_cov[0] + '.nc')
    lat = xr.open_dataset(dir + hr_cov[1] + '.nc')
    lon = xr.open_dataset(dir + hr_cov[2] + '.nc')
    coast = xr.open_dataset(dir + hr_cov[3] + '.nc')

    prism_coarse = xr.open_mfdataset(dir + var + '_' + months[i] + '_' + hr[0] + '_coarse.nc')
    
    # standardize variables 

    # writing standardization data to csv file to unstandardize after predicting 
    mydict =[{'var': 'prec','clim': 'prism', 'mean': prism_mean, 'std':prism_std}, 
            {'var': 'prec', 'clim': 'worldclim', 'mean': worldclim_mean, 'std':worldclim_std}, 
            {'var': 'dist from coast', 'clim': 'prism', 'mean':coast_mean, 'std':coast_std},
            {'var': 'dem', 'clim': 'prism', 'mean':dem_mean, 'std':dem_std},
            {'var': 'lat', 'clim': 'prism', 'mean':lat_mean, 'std':lat_std},
            {'var': 'lon', 'clim': 'prism', 'mean':lon_mean, 'std':lon_std},
            ]

    # field names 
    fields = ['var', 'clim', 'mean', 'std'] 

    with open('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/standardization.csv', 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)

        writer.writeheader() 

        writer.writerows(mydict)

    # write to netcdf 

    prism.prec.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/prism_train.nc')
    worldclim.prec.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/worldclim_train.nc')
    dem.dem.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/dem_train.nc')
    lat.lat.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/lat_train.nc')
    lon.lon.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/lon_train.nc')
    coast.dist.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/coast_train.nc')

    prism_test.prec.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/prism_test.nc')
    worldclim_test.prec.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/worldclim_test.nc')
    dem_test.dem.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/dem_test.nc')
    lat_test.lat.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/lat_test.nc')
    lon_test.lon.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/lon_test.nc')
    coast_test.dist.to_netcdf('C:/Users/SBEALE/Desktop/GAN Data/prec/worldclim/march_nonan_bc_replace_w_WRF/coast_test.nc')
        
    # make tiles 
    
    # remove tiles with nan 
    
    # save pt files to directory 
    
        size_fine = 128*128
size_coarse = 32*32

