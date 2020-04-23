# CONVERT CMAQ NETCDF OUTPUT FILE TO RASTER / GEOTIF / SHAPEFILE

import rioxarray
import xarray
import numpy as np

d = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/o3/'

dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 
ll = xarray.open_dataset(dir+ ll)

fnames = ['COMBINE_ACONC_201808.nc',  'dailymaxozone_201808.nc',  'NO2_201808.nc ', 'O3_201808.nc',  'pm25_201808.nc', 'COMBINE_ACONC_201901.nc',  'dailymaxozone_201901.nc',  'NO2_201901.nc',  'O3_201901.nc',  'pm25_201901.nc']


#  ------- DOING AUG O3

fnames_out = ['dailymaxozone_201808'+str(i+1).zfill(2)+'.tif' for i in range(31)] #set up names for files out

for i in range(31): # number of days in these files
   xds = xarray.open_dataset(d+ 'dailymaxozone_201808.nc')
   # I'm writing out each time step as its own file name and only taking the first layer
   # hence taking the ith time step and 0th layer
   data = xds["O3"][i][0] 
   # Key here is literally spelling out the indices and the coordinates with x, y 
   foo=xarray.DataArray(data, coords={"x": np.arange(0,len(ll.lat)),"y": np.arange(0,len(ll.lat[0])),"latitude": (["x","y"],ll.lat),"longitude": (["x","y"],ll.lon)},dims=["x","y"])
   # and this is how you write out the file
   foo.T.rio.to_raster(fnames_out[i])

# ------- DOING JAN O3

fnames_out = ['dailymaxozone_201901'+str(i+1).zfill(2)+'.tif' for i in range(31)]

for i in range(31): # number of days in these files
   xds = xarray.open_dataset(d+ 'dailymaxozone_201901.nc')
   data = xds["O3"][i][0]
   foo=xarray.DataArray(data, coords={"x": np.arange(0,len(ll.lat)),"y": np.arange(0,len(ll.lat[0])),"latitude": (["x","y"],ll.lat),"longitude": (["x","y"],ll.lon)},dims=["x","y"])
   foo.T.rio.to_raster(fnames_out[i])

# ------- DOING JAN O3

fnames_out = ['dailymaxozone_201901'+str(i+1).zfill(2)+'.tif' for i in range(31)]

for i in range(31): # number of days in these files
   xds = xarray.open_dataset(d+ 'dailymaxozone_201901.nc')
   data = xds["O3"][0][0] # theres only 1 timestep here
   foo=xarray.DataArray(data, coords={"x": np.arange(0,len(ll.lat)),"y": np.arange(0,len(ll.lat[0])),"latitude": (["x","y"],ll.lat),"longitude": (["x","y"],ll.lon)},dims=["x","y"])
   foo.T.rio.to_raster(fnames_out[i])

# ------- AVERAGE O3

fnames_in = ['O3_201808.nc','O3_201901.nc' ]
fnames_out = ['O3_201808.tif','O3_201901.tif' ]

for i in range(len(fnames_in)): # number of days in these files
   xds = xarray.open_dataset(d+ fnames_in[i])
   data = xds["O3"][0][0]
   foo=xarray.DataArray(data, coords={"x": np.arange(0,len(ll.lat)),"y": np.arange(0,len(ll.lat[0])),"latitude": (["x","y"],ll.lat),"longitude": (["x","y"],ll.lon)},dims=["x","y"])
   foo.T.rio.to_raster(fnames_out[i])


# ------- DOING NO2

fnames_in = ['NO2_201808.nc','NO2_201901.nc' ]
fnames_out = ['NO2_201808.tif','NO2_201901.tif' ]

for i in range(len(fnames_in)): # number of days in these files
   xds = xarray.open_dataset(d+ fnames_in[i])
   data = xds["NO2"][0][0]
   foo=xarray.DataArray(data, coords={"x": np.arange(0,len(ll.lat)),"y": np.arange(0,len(ll.lat[0])),"latitude": (["x","y"],ll.lat),"longitude": (["x","y"],ll.lon)},dims=["x","y"])
   foo.T.rio.to_raster(fnames_out[i])
