from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from scipy import stats
import datetime

def slice_time(forecast_ds, reanalysis_ds):
    """
    returns slices of forecast_ds and reanalysis_ds so that
    the slices are over the same time range (intersection of the two)

    inputs
    -------

           forecast_ds     (xarray Dataset) : T * lat * lon array of 2m temperature
           reanalysis_ds   (xarray Dataset) : T * lat * lon array of 2m temperature

    ouputs
    ------

           2 datasets (sliced_forecast_ds, sliced_reanalysis_ds)
    """
    first_forecast_dt = forecast_ds['time'][0].values

    last_forecast_dt = forecast_ds['time'][-1].values


    first_reanalysis_dt = reanalysis_ds['time'][0].values

    last_reanalysis_dt = reanalysis_ds['time'][-1].values

    first_index = max(first_forecast_dt, first_reanalysis_dt)
    last_index = min(last_forecast_dt, last_reanalysis_dt)

    sliced_forecast_ds = forecast_ds.sel(time=slice(first_index, last_index))
    sliced_reanalysis_ds = reanalysis_ds.sel(time=slice(first_index, last_index))

    return sliced_forecast_ds, sliced_reanalysis_ds

def shift_by_lead_time(forecast_ds, lead_time):
    """
    returns forecast_ds where the time dimension
    is shifted by lead_time

    inputs
    -------

           forecast_ds    (xarray Dataset) : T * lat * lon array of 2m temperature
           lead_time      (int)            : the leadtime of the forecast (eg 1 indicates a 1 month forecast)

    ouputs
    ------


    """

    first_dt = pd.to_datetime(forecast_ds['time'].values[0]).replace(day=1)

    shifted_start_time = first_dt + pd.tseries.offsets.DateOffset(months=lead_time)
    #offseting forecast time dimension according to lead time

    return pd.date_range(shifted_start_time, periods=len(forecast_ds['time'].values), freq=pd.offsets.MonthBegin(1))
    #creating date range where every month in the range begins on the first of the month

def remove_incomplete_seasons(dataset):
    """
    returns a data array where incomplete seasons are removed

    inputs
    -------

           dataset    (xarray Dataset) : T * lat * lon array of 2m temperature

    ouputs
    ------

           dataset (xarray)
    """
    time = dataset['time.season'].values
    #grabbing all seasons from the data_array
    len_time = len(time)

    def remove_start(seasons):
        if seasons[0] != seasons[1]:
            return 1
        elif seasons[0] != seasons[2]:
            return 2
        else:
            return 0

    def remove_end(seasons):
        if seasons[-1] != seasons[-2]:
            return -1
        elif seasons[-1] != seasons[-3] :
            return -2
        else:
            return len_time

    start_index = remove_start(time)
    #returns the index of the first month of the first complete season
    end_index = remove_end(time)
    #returns the index of the last month of the last complete season
    slice_of_dataset = dataset.isel(time=slice(start_index, end_index))
    #slices the data_array to only grab complete seasons

    assert len(slice_of_dataset['time']) % 3 == 0, "Slicing did not occur properly"
    return slice_of_dataset

def regrid(forecast_ds, reanalysis_ds, regrid_out_ds):
    """
    Returns 2 data arrays corresponding to the input paths which are cleaned and regridded

    inputs
    -------

           forecast_dataset          (xarray dataset) : the forecast data
           reanalysis_dataset        (xarray dataset) : the reanalysis data
           regrid_out                (xarray dataset) : the dummy grid for regridding

    ouputs
    ------

           2 xarray data arrays : T * lat * lon of 2m temperature
    """
    forecast_regridder = xe.Regridder(forecast_ds, regrid_out_ds, 'nearest_s2d', reuse_weights=True)
    #building forecast regridder
    forecast_data_array= forecast_regridder(forecast_ds['2t'][:])
    #regridding forecast 2m temp

    reanalysis_regridder = xe.Regridder(reanalysis_ds, regrid_out_ds, 'nearest_s2d', reuse_weights=True)
    #building reanalysis regridder

    reanalysis_data_array = reanalysis_regridder(reanalysis_ds['2t'][:])
    #regridding reanalysis 2m temp

    return forecast_data_array, reanalysis_data_array


def preprocess_forecast_and_reanalysis(forecast_ds,
                                      reanalysis_ds,
                                      forecast_type,
                                      reanalysis_type,
                                      forecast_start_time,
                                      reanalysis_start_time,
                                      frequency,
                                      lead_time=0):
    """
    cleans the forecast and reanalysis datasets by renaming dimensions,
    converting to celsius (if necessary) and by squeezing extra dimensions

    inputs
    -------

           forecast_ds           (xarray dataset) : dataset of 2m temperature
           reanalysis_ds         (xarray dataset) : dataset of 2m temperature
           forecast_type         (str)            : string of the forecast type
           reanalysis_type       (str)            : string of the reanalysis type
           forecast_start_time   (str)            : string in form 'YYYY-MM-DD'
           reanalysis_start_time (str)            : string in form 'YYYY-MM-DD'
           frequency             (str)            : python Timestamp frequency
           lead_time             (int)            : lead time of forecast


    ouputs
    ------

           2 xarray data arrays
    """
    assert forecast_type in ['CFS', 'NMME', 'SEAS'], "Forecast model type not valid, enter either 'CFS', 'NMME' or 'SEAS'"
    assert reanalysis_type in ['NCEP', 'Berkeley', 'ERA'], "Reanalysis data type not valid, enter either 'NCEP' or 'Berkeley'"

    first_forecast_dt = pd.to_datetime(forecast_start_time).replace(day=1, hour=0)
    #replacing the day and hour value of the timestamp to the first day/hour
    #so that the range begins on the correct month

    forecast_ds['time'] = pd.date_range(first_forecast_dt, periods=len(forecast_ds['time']), freq=frequency)
    #creating list of pandas timestamp objects

    first_reanalysis_dt = pd.to_datetime(reanalysis_start_time).replace(day=1, hour=0)
    #replacing the day and hour value of the timestamp to the first day/hour
    #so that the range begins on the correct month

    reanalysis_ds['time'] = pd.date_range(first_reanalysis_dt, periods=len(reanalysis_ds['time']), freq=frequency)
    #creating list of pandas timestamp objects


    forecast_ds['time'] = shift_by_lead_time(forecast_ds, lead_time)
    #pushes forward forecast_ds time dimension to reflect the forecast
    #duration from the original start date.

    sliced_forecast_ds, sliced_reanalysis_ds = slice_time(forecast_ds, reanalysis_ds)
    #slices forecast and reanalysis data according to the time range of the forecast data

    if reanalysis_type == 'Berkeley':

        sliced_reanalysis_ds.rename({'latitude': 'lat', 'longitude': 'lon', 'temperature': '2t'}, inplace=True)
        #renames the reanalysis_ds to standard naming convention. Air is 2m temp

    elif reanalysis_type == 'NCEP':
        sliced_reanalysis_ds = sliced_reanalysis_ds.squeeze('level')
        sliced_reanalysis_ds = sliced_reanalysis_ds.drop('level')
        sliced_reanalysis_ds.rename({'air': '2t'}, inplace=True)
        #removes extraneous dimension from NCEP

    elif reanalysis_type == 'ERA':
        sliced_reanalysis_ds.rename({'latitude': 'lat', 'longitude': 'lon', 't2m': '2t'}, inplace=True)

    if forecast_type in ['CFS', 'NMME']:
        sliced_forecast_ds = sliced_forecast_ds.squeeze('height')
        sliced_forecast_ds = sliced_forecast_ds.drop('height')
        #removes extraneous dimension from cvfsv2

    return sliced_forecast_ds, sliced_reanalysis_ds
    
def anomalize(data_array):

    """
    anomalizes the data_array and returns a new data array
    and the climatology

    inputs
    -------

           data_array          (xarray data array) : data array of 2m temperature

    ouputs
    ------

           anomalized data array, climatology data array (xarrays)
    """

    monthly_climatology = data_array.groupby('time.month').mean('time')

    assert monthly_climatology.shape[0] == 12, "Output dimension is not correct"
    # finds the mean temperature per lat/lon point by month
    anomalized_monthly = data_array - monthly_climatology[data_array['time.month'] - 1]
    #anomalizes the data_array by the corrsponding month

    return anomalized_monthly

def convert_from_monthly_to_seasonal(data_array):

    cleaned_seasons_data_array = remove_incomplete_seasons(data_array)

    seasonal_data_array = cleaned_seasons_data_array.resample(time='3MS').mean(axis=0)

    assert len(seasonal_data_array['time']) * 3 == len(cleaned_seasons_data_array['time']), "Grouping of seasons did not occur correctly"

    return seasonal_data_array

def select_season_or_month(data_array, mode, time):

    def select_month(data_array, month):
        return data_array[data_array['time.month'] == month]

    def select_season(data_array, season):
        return data_array[data_array['time.season'] == season]

    assert mode == 'seasonal' or mode == 'monthly', 'Mode not valid, choose either seasonal or monthly'

    if (mode == 'seasonal') :

        seasons_list = ['DJF', 'MAM', 'JJA', 'SON']

        assert time in seasons_list, 'Time selected is not a valid season'

        return select_season(data_array, time)

    elif (mode == 'monthly'):

        time = int(time)

        months_list = np.arange(1,13,1)

        assert time in months_list, 'Time selected is not a valid month'

        return select_month(data_array, time)

def read_in_dataset_and_anomalize(forecast_dataset,
                                  reanalysis_dataset,
                                  regrid_dataset,
                                  forecast_type,
                                  reanalysis_type,
                                  forecast_start_time,
                                  reanalysis_start_time,
                                  frequency,
                                  mode='monthly',
                                  time=None,
                                  lead_time=0):

    """
    combines preprocess_forecast_and_reanalysis, load_data_and_regrid,
    and select_time_and_anomalize functions so that one can enter the
    forecast and reanalysis datasets along with the other required arguments
    and returns the corresponding data arrays of anomalized 2m temperature data.

    inputs
    -------

            forecast_dataset          (xarray dataset) : the forecast data
            reanalysis_dataset        (xarray dataset) : the reanalysis data
            regrid_dataset             (xarray dataset) : the dummy grid for regridding
            forecast_type             (str)            : string of the forecast type
            reanalysis_type           (str)            : string of the reanalysis type
            mode                      (str)            : the mode for how to group the data
            time                      (str)            : string for the season or month to select
            lead_time                 (int)            : number of months to shift forward the forecast_dataset

    ouputs
    ------

           2 xarray data arrays that have been preprocessed, regridded, and anomalized
           2 xarrays of climatology for forecast and reanalysis respectively
    """
    preprocessed_forecast_dataset, preprocessed_reanalysis_dataset = preprocess_forecast_and_reanalysis(forecast_dataset,
                                                                                                        reanalysis_dataset,
                                                                                                        forecast_type,
                                                                                                        reanalysis_type,
                                                                                                        forecast_start_time,
                                                                                                        reanalysis_start_time,
                                                                                                        frequency,
                                                                                                        lead_time)
    #preprocessing datasets to standardize naming conventions

    forecast_temp_da, reanalysis_temp_da = regrid(preprocessed_forecast_dataset,
                                                  preprocessed_reanalysis_dataset,
                                                  regrid_dataset,
                                                  )
    #regrids data and bounds them to have the same time dimension

    anomalized_forecast_temp_da = anomalize(forecast_temp_da)
    #anomalizes forecast data and groups by mode, then selects by time

    anomalized_reanalysis_temp_da = anomalize(reanalysis_temp_da)
    #anomalizes reanalysis data and groups by mode, then selects by time

    if mode == 'seasonal':
        anomalized_forecast_temp_da = convert_from_monthly_to_seasonal(anomalized_forecast_temp_da)
        anomalized_reanalysis_temp_da = convert_from_monthly_to_seasonal(anomalized_reanalysis_temp_da)

    if time != None:
        anomalized_forecast_temp_da = select_season_or_month(anomalized_forecast_temp_da, mode, time)
        anomalized_reanalysis_temp_da = select_season_or_month(anomalized_reanalysis_temp_da, mode, time)

    return anomalized_forecast_temp_da, anomalized_reanalysis_temp_da

def find_intersections_of_da_list(da_list):

    if len(da_list) == 3:
        A, B, C = da_list[0], da_list[1], da_list[2]

        sliced_A, sliced_B = slice_time(A, B)
        #intersection of A and B
        re_sliced_B, sliced_C = slice_time(sliced_B, C)
        #intersection of modified B and C
        re_sliced_A, re_sliced_C = slice_time(sliced_A, sliced_C)
        #intersection of modified A and modified C

        return [re_sliced_A, re_sliced_B, re_sliced_C]

    else:
        A, B = da_list[0], da_list[1]

        sliced_A, sliced_B = slice_time(A, B)

        returrn [sliced_A, sliced_B]

def read_in_and_anomalize_multiple_datasets(ds_dict,
                                            dict_type,
                                            constant_dataset,
                                            main_grid,
                                            constant_type,
                                            constant_start_time,
                                            frequency,
                                            mode,
                                            time,
                                            lead_time):
    forecast_list = []
    reanalysis_list = []

    for ds_type in ds_dict:

        if dict_type == 'forecast':

            forecast_dataset = ds_dict[ds_type][0]
            forecast_type = ds_type
            forecast_start_time = ds_dict[ds_type][1]

            reanalysis_dataset = constant_dataset
            reanalysis_type = constant_type
            reanalysis_start_time = constant_start_time

        elif dict_type == 'reanalysis':
            forecast_dataset = constant_dataset
            forecast_type = constant_type
            forecast_start_time = constant_start_time

            reanalysis_dataset = ds_dict[ds_type][0]
            reanalysis_type = ds_type
            reanalysis_start_time = ds_dict[ds_type][1]

        forecast_da, reanalysis_da = read_in_dataset_and_anomalize(forecast_dataset,
                                                                reanalysis_dataset,
                                                                main_grid,
                                                                forecast_type,
                                                                reanalysis_type,
                                                                forecast_start_time,
                                                                reanalysis_start_time,
                                                                frequency,
                                                                mode,
                                                                time,
                                                                lead_time)
        forecast_list.append(forecast_da)
        reanalysis_list.append(reanalysis_da)


    sliced_forecast_list = find_intersections_of_da_list(forecast_list)
    sliced_reanalysis_list = find_intersections_of_da_list(reanalysis_list)

    if dict_type == 'forecast':

        forecast_keys = [*ds_dict]
        forecast_da_dict = {}

        for i in range(len(sliced_forecast_list)):
            forecast_da_dict[forecast_keys[i]] = sliced_forecast_list[i]

        return forecast_da_dict, sliced_reanalysis_list[0]

    elif dict_type == 'reanalysis':

        reanalysis_keys = [*ds_dict]
        reanalysis_da_dict = {}

        for i in range(len(sliced_reanalysis_list)):
            reanalysis_da_dict[reanalysis_keys[i]] = sliced_reanalysis_list[i]

        return sliced_forecast_list[0], reanalysis_da_dict

def find_corr(forecast_data_array, reanalysis_data_array):
    """
    finds correlation between forecast and reanalysis data arrays

    inputs
    -------

           data_array        (xarray data array) : T * lat * lon array of 2m temperature

    ouputs
    ------

           xarray dataarray of shape(T, lat, lon)
    """
    mean_reanalysis = np.mean(reanalysis_data_array, axis=0)
    mean_forecast = np.mean(forecast_data_array, axis=0)
    e_xy = np.mean( reanalysis_data_array * forecast_data_array, axis=0)
    return (e_xy - (mean_reanalysis*mean_forecast))/np.sqrt(np.var(forecast_data_array,axis=0)*np.var(reanalysis_data_array,axis=0))

def get_corr_lat_lon(correlation_data_array, latitude, longitude):
    lats = correlation_data_array['lat']
    lons = correlation_data_array['lon']
    nearest_lat = np.asscalar(np.abs(lats - latitude).argmin()) # get nearest latitude index
    nearest_lon = np.asscalar(np.abs(lons - longitude).argmin()) # get nearest longitude index
    correlation = correlation_data_array[nearest_lat, nearest_lon]
    return correlation

def extract_timeseries_at_lat_lon(data_array, latitude, longitude):
    lats = data_array['lat']
    lons = data_array['lon']
    nearest_lat = np.asscalar(np.abs(lats - latitude).argmin()) # get nearest latitude index
    nearest_lon = np.asscalar(np.abs(lons - longitude).argmin()) # get nearest longitude index
    timeseries_at_lat_lon = data_array[:, nearest_lat, nearest_lon]
    return timeseries_at_lat_lon

def find_rmse(data_1, data_2, ax=0):
    """
    Finds RMSE between data_1 and data_2

    Inputs
    ------

          data_1 (np.array)
          data_2 (np.array)
          ax     (int) The axis (or axes) to mean over


    Outpts
    ------

          (int) RMSE between data_1 and data_2
    """
    return np.sqrt(np.mean((data_1 - data_2)**2, axis=ax))


def find_weighted_rmse(data_1, data_2, weights, axes=0):

    unweighted_rmse = find_rmse(data_1, data_2, axes)

    return np.mean((unweighted_rmse * weights)/np.mean(weights))

def find_weighted_correlation(data_array_1, data_array_2, weights, axes=0):
    unweighted_corr = find_corr(data_array_1, data_array_2)
    return np.mean((unweighted_corr * weights)/np.mean(weights))


def plot_global(img_array, lats, lons, title, cbar_label,
               vmin, vmax, cmap='bwr', out_path=None):
    """
    inputs
    ------

        img_array  (np arr) : the field you wish to plot (i.e. tas, slp)
        lats       (np arr) : np array of lats
        lons       (np arr) : np array of lons
        title      (str)    : the string corresponding to the title
        cbar_label (str)    : the label of the colorbar
        out_path   (str)    : the path at which to save the figure
        vmin       (float)    : the value that will correspond to the min color of the colormap
        vmax       (float)    : the value that will correspond to the max color of the colormap

    outputs
    -------

        None, but with the %matplotlib inline command, this function
        will show the contourf plot with the corresponding
    """
    plt.figure(figsize=(14,7))
    m = Basemap(projection='cyl', llcrnrlat=min(lats), lon_0=np.median(lons),
        llcrnrlon=min(lons), urcrnrlat=max(lats), urcrnrlon=max(lons), resolution = 'c')

    xx, yy = np.meshgrid(lons, lats)
    x_map,y_map = m(xx,yy)
    m.contourf(x_map,y_map,img_array,256,cmap=cmap, vmin=vmin,
              vmax=vmax)
    cbar = m.colorbar()
    cbar.ax.set_ylabel(cbar_label)
    m.drawcoastlines()
    plt.title(title)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
        plt.show()
    plt.clf()
    plt.close()
