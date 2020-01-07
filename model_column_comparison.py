#!/bin/python

#model to column comparison
#---------------------------------------


#------------------------------------------
# Libraries
#--------------
from matplotlib import pyplot as plt ; from matplotlib import colors
import numpy as np; import numpy.ma as ma; from matplotlib.patches import Path, PathPatch
import pandas as pd; from shapely.geometry import Point, shape, Polygon;import fiona
from shapely.ops import unary_union, cascaded_union; from geopandas.tools import sjoin
import geopandas as gpd; import geoplot; import glob; import os; from datetime import timedelta, date;
from netCDF4 import Dataset
import scipy.ndimage; from cartopy import crs as ccrs; from cartopy.io.shapereader import Reader
import matplotlib.path as mpath; import seaborn as sns
#------------------------------------------

column_dir = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/column/'
onlyfiles = next(os.walk(column_dir))[2]
onlyfiles.sort() 

fig_dir = '~/figs_for_dan/'

path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)

#get lat lon from grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 
llx=Dataset(dir+ll,'r')
lat,lon=llx['lat'][:],llx['lon'][:]

model_columns = np.asarray([Dataset(column_dir + onlyfiles[i],'r')['NO2'] for i in range(len(onlyfiles))])

model_columns_month = np.array([model_columns[i].mean(axis=0) for i in range(len(model_columns))]).mean(axis=0)
#model_columns_no2 = model_columns[i]['NO2']

model_columns_month_ish= model_columns_month*10**-20

sat_columns = pd.read_csv('/projects/b1045/NO2/l3/nitrogendioxide_tropospheric_column.csv',index_col = 0)

sat_columns = sat_columns*10**5
model_columns_month_ish = model_columns_month_ish*10**5

# Make scatter
#--------------------------------

plt.scatter(sat_columns, model_columns_month_ish, alpha = 0.84, color = 'purple')
plt.scatter(sat_columns, model_columns_month_ish, alpha = 0.84, color = 'purple')
plt.xlim(0,0.00021*10**5)
plt.ylim(0,0.00021*10**5)
plt.xlabel('TropOMI Column ('+ Dataset(column_dir + onlyfiles[i],'r')['NO2'].units + '*10^15)')
plt.ylabel('CMAQ Column')
plt.plot([-100,100],[-100,100],c='black',alpha = 0.5)

from scipy.stats import pearsonr
sat_columns  = np.array(sat_columns) 
scr, mcr = sat_columns.ravel(), model_columns_month_ish.ravel()
bad = np.isnan(scr)

r = round(pearsonr(mcr[~bad],scr[~bad])[0],2)

plt.title('R = '+ str(r))

plt.savefig(fig_dir+'sat_to_model.png')

fig,ax = plt.subplots(figsize = (6,6))
from palettable.colorbrewer.sequential import OrRd_4

[plt.scatter(stn_epa_mix[i]['Sample Measurement'], pd.DataFrame(stnpixel_from_cmaq)[i], alpha = 0.99, color = OrRd_4.mpl_colors[i+1]) for i in range(len(epa_lat))]
plt.xlim(0,27)
plt.ylim(0,27)
plt.xlabel('EPA Station (ppb)')
plt.ylabel('CMAQ (ppb)')
plt.plot([-100,100],[-100,100],c='black',alpha = 0.5)

ab = np.array([np.array(stn_epa_mix[i]['Sample Measurement']) for i in range(len(stn_epa_mix))]).ravel()
ba = np.array([np.array(pd.DataFrame(stnpixel_from_cmaq)[i]) for i in range(len(stn_epa_mix))]).ravel()

bad = np.isnan(ab)

r = round(pearsonr(ab[~bad],ba[~bad])[0],2)

plt.title('Daily Average R = '+ str(r))

plt.savefig(fig_dir+'stn_to_model.png')

# make MAPS 
#--------------------------------

#options
crs_new = ccrs.PlateCarree()
import cartopy.feature as cfeature 
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE

vmin,vmax = 2, 20
levels = np.arange(vmin, vmax, (vmax-vmin)/10)
cmap = 'magma_r'
xl,xu,yl,yu = lon.min()+1,lon.max()-1,lat.min()+1,lat.max()-1
xl,xu,yl,yu = lon.ravel()[~bad].min(),lon.ravel()[~bad].max(),lat.ravel()[~bad].min(),lat.ravel()[~bad].max()

# sat column
fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 8))
cs = ax.pcolormesh(lon,lat, sat_columns,transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.5)
cbar.set_ticks(levels)
cbar.set_ticks(levels)
ax.set_extent([xl,xu,yl,yu])
ax.set_title('Regridded TropOMI NO2')
states = cfeature.STATES.with_scale('10m')
ax.add_feature(states)

plt.savefig(fig_dir+'sat_no2.png')

# model column
fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 8))
cs = ax.pcolormesh(lon,lat, model_columns_month_ish[0],transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.5)
cbar.set_ticks(levels)
ax.set_extent([xl,xu,yl,yu], crs= crs_new)
ax.set_title('CMAQ Column NO2')
states = cfeature.STATES.with_scale('10m')
ax.add_feature(states)
plt.savefig(fig_dir+'model_no2.png')


#difference bw model and satellite
from palettable.colorbrewer.diverging import RdGy_10

difference = model_columns_month_ish[0] - sat_columns
vmin,vmax = difference[~np.isnan(difference)].min(),difference[~np.isnan(difference)].max()
vmin,vmax = -8,8
cmap = RdGy_10.mpl_colormap
levels = np.arange(vmin, vmax, (vmax-vmin)/10)

fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 8))
cs = ax.pcolormesh(lon,lat, difference,transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.5)
cbar.set_ticks(levels)
ax.set_extent([xl,xu,yl,yu])
states = cfeature.STATES.with_scale('10m')
ax.add_feature(states)
ax.set_title('Delta Column NO2')
plt.savefig(fig_dir+'difference_no2.png')


# timeseries_model

model_columns = np.asarray([Dataset(column_dir + onlyfiles[i],'r')['NO2'] for i in range(len(onlyfiles))])
model_columns_month = np.array([model_columns[i].mean(axis=0) for i in range(len(model_columns))])
model_columns_month = model_columns_month*10**-15
cmap = 'magma_r'
levels = np.arange(vmin, vmax, (vmax-vmin)/10)
xl,xu,yl,yu = lon.min()+1,lon.max()-1,lat.min()+1,lat.max()-1

vmin,vmax = 2, 20

for i in range(len(model_columns_month)):
# model column
   fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(10, 6))
   cs = ax.pcolormesh(lon,lat, model_columns_month[i][0],transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
   cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.5)
   cbar.set_ticks(levels)
   ax.set_extent([xl,xu,yl,yu], crs= crs_new)
   ax.set_title('CMAQ Column NO2')
   states = cfeature.STATES.with_scale('10m')
   ax.add_feature(states)
   plt.savefig(fig_dir+'timeseries_model_column'+str(i)+'.png')
   plt.close()

vmin,vmax = 2, 20

path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])

for i in range(5):
   for j in range(len(model_columns[0])):
#for i in range(1):
#   for j in range(1):
      data = model_columns[i][j][0]*10**-15
      fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(10, 6))
      #ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
      cs = ax.pcolormesh(lon,lat, data,transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
      cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.5)
      cbar.set_ticks(levels)
      ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
      x=[chi_shapefile.bounds.minx.min(), chi_shapefile.bounds.maxx.max()] 
      y=[chi_shapefile.bounds.miny.min(), chi_shapefile.bounds.maxy.max()]
      #ax.set_extent([x[0]-.03,x[1]+.03,y[0]-.03,y[1]+.03],crs= crs_new)
      ax.set_extent([xl,xu,yl,yu], crs= crs_new)
      ax.set_title('CMAQ Column')
      states = cfeature.STATES.with_scale('10m')
      ax.add_feature(states)
      plt.savefig(fig_dir+'timeseries_model_column_d'+str(i)+'_h'+str(j)+'.png')
      plt.close()



