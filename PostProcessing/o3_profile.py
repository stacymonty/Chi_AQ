#!/usr/bin/env python

# o3_column_june2021.py


# ---------------------------------------------------------------------
# Stacy Montgomery, NOV 2018 - DEC 2018
# Plot o3 column over Chicago to watch how it transitions
# ---------------------------------------------------------------------
#                             USER INPUT
# ---------------------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import netCDF4
import math
from scipy.interpolate import griddata
import scipy.stats as st
import cartopy.feature as cfeature 
from cartopy import crs as ccrs;
from shapely.ops import unary_union, cascaded_union
from geopandas.tools import sjoin
from shapely.geometry import Point, shape
from cartopy import crs as ccrs;
import geopandas as gpd
import moviepy.editor as mpy
import os
import glob
import pandas as pd; from shapely.geometry import Point, shape, Polygon;import fiona
from shapely.ops import unary_union, cascaded_union; from geopandas.tools import sjoin
import matplotlib.path as mpath;
from cartopy.io.shapereader import Reader

import matplotlib.colors as colors

# ---------------------------------------------------------------------

# dir to grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 

dir='/home/asm0384/'
ll = 'lat_lon_chicago_d02.nc'

dir_cmaq_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'
dir_cmaq_d03_wint='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/'

dir_cmaq_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/'
dir_cmaq_d02_wint='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/'


#names of lat lons in the cmaq grid
la,lo='lat','lon' # for 1.3km

# 
year='2018'
month='8'
ssn = 'Summer'

startswith = 'CCTM_CONC'

# shapes and directories == https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=State+Legislative+Districts
path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])

# ---------------------------------------------------------------------

def pull_cmaq(dir_CMAQ,startswith,cmaq_var):
   #pull model files from given directoy
   onlyfiles = next(os.walk(dir_CMAQ))[2]
   onlyfiles.sort() # so that searching for dates are easier
   # pull only CONC files
   fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
   print(fnames_CMAQ)
   # get data files
   ncfile_CMAQ_base = [Dataset(dir_CMAQ+ fnames_CMAQ[i],'r') for i in range(len(fnames_CMAQ))]
   return ncfile_CMAQ_base



def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
# stn -- list (points) 
# wrf -- list (grid)
   xx=[];yy=[]
   for i in range(len(stn_lat)):
      abslat = np.abs(wrf_lat-stn_lat[i])
      abslon= np.abs(wrf_lon-stn_lon[i])
      c = np.maximum(abslon,abslat)
      latlon_idx = np.argmin(c)
      x, y = np.where(c == np.min(c))
      #add indices of nearest wrf point station
      xx.append(x) 
      yy.append(y)
   #return indices list
   return xx, yy


def mask_given_shapefile(lon,lat,shapefile):
   '''
   Make a mask given a shapefile
   lon - array of grid lons
   lat - array of grid lats
   shapefile - geopandas geodataframe shapefile
   '''
   union=gpd.GeoSeries(unary_union(shapefile.geometry))
   mask=np.ones(lon.shape,dtype=bool)
   mask[:] = False
   for i in range(len(lon)):
       for j in range(len(lon[0])):
          pt = Point(lon[i][j],lat[i][j])
          mask[i][j] =  pt.within(union[0])
   #
   return mask

# ---------------------------------------------------------------------



# I think  day 10 is best
# what day is best day for pbl fomation
#fig,ax = plt.subplots(10,3,figsize=(11,11))
#count = 0

#for day in range(21,30):
# 	sli= [np.array([np.array(base[day]['O3'][t][l][xl]).T[yl:yu+3] for l in range(35)]) for t in range(24)]
# 	utc = 6
# 	sli_morn = np.mean(sli[7+utc:10+utc],axis=0)
# 	sli_mid = np.mean(sli[11+utc:14+utc],axis=0)
# 	sli_after =  np.mean(sli[15+utc:18+utc],axis=0)
# 	#fig,ax = plt.subplots(1,3)
# 	ax[count][0].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,len(sli[0])),sli_morn,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
# 	ax[count][1].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,len(sli[0])),sli_mid,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
# 	ax[count][2].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,len(sli[0])),sli_after,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
# #	count = count+1

#plt.tight_layout()
#plt.savefig('pbl_o3_21-30.png')

# Make average Chicago slice~
# chicago box: upper lat lon
#lolo = -87.939930; lola= 41.644543 
#ulo = -87.524137; ula = 42.023039
#xu,yu = find_index([ulo],[ula],lon,lat)
#xl,yl = find_index([lolo],[lola],lon,lat)
#xu,yu,xl,yl = xu[0]+4,yu[0]+4,xl[0]-4,yl[0]-4
#
# adjustable plotting parts
llx=Dataset(dir+ll,'r')
#lat,lon=llx['lat'][:],llx['lon'][:]
lat,lon=llx['LAT'][0][0],llx['LON'][0][0]


#mask = mask_given_shapefile(lon,lat,chi_shapefile)
base = pull_cmaq(dir_cmaq_d02,startswith,"O3")

# Pull single row from data
la = 41.8
lolo = -87.939930; ulo = -87.524137;
xu,yu = find_index([ulo],[la],lon,lat)
xl,yl = find_index([lolo],[la],lon,lat)

xu,yu,xl,yl  = xu[0][0],yu[0][0],xl[0][0]+2,yl[0][0]

# # check where we're plotting
crs_new = ccrs.PlateCarree()
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(8, 6))
#axs.scatter(lon[xl:xu].T[yl:yu],lat[xl:xu].T[yl:yu])
#axs.scatter(lon[xl].T[yl:yu+3],lat[xl].T[yl:yu+3])
#axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
chi_shapefile.plot(ax=axs,facecolor="None")
axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
axs.plot(lon[xl].T[yl:yu+3],lat[xl].T[yl:yu+3])
plt.show()

#morning = [7,8,9,10]
#midday = [12,13,14]
#after = [16,17,18]


# make slices of 2 rep days
# ozone profiles
utc = 6
day = 9


base = pull_cmaq(dir_cmaq_d02,startswith,"O3")
sli= [np.array([np.array(base[day]['NO2'][t][l][xl]).T[yl:yu+3] for l in range(35)]) for t in range(24)]
sli_morn = np.mean(sli[7+utc:10+utc],axis=0)*1000
sli_mid = np.mean(sli[11+utc:14+utc],axis=0)*1000
sli_after =  np.mean(sli[16+utc:17+utc],axis=0)*1000

del base
del sli

wbase = pull_cmaq(dir_cmaq_d02_wint,startswith,"O3")
wbase = wbase[10:]
wint_sli= [np.array([np.array(wbase[day]["NO2"][t][l][xl]).T[yl:yu+3] for l in range(35)]) for t in range(24)]
wsli_morn = np.mean(wint_sli[7+utc:10+utc],axis=0)*1000
wsli_mid = np.mean(wint_sli[11+utc:14+utc],axis=0)*1000
wsli_after =  np.mean(wint_sli[16+utc:17+utc],axis=0)*1000

del wbase
del wint_sli


vmin = 0
vmax = 30


cmap = 'Purples'
fig,ax = plt.subplots(2,3,figsize = (10,7))
ax=ax.ravel()
#ax[0].pcolormesh(lon[xl].T[yl:yu],np.arange(0,len(sli[0])),sli_morn,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
#ax[1].pcolormesh(lon[xl].T[yl:yu],np.arange(0,len(sli[0])),sli_mid,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
#cs = ax[2].pcolormesh(lon[xl].T[yl:yu],np.arange(0,len(sli[0])),sli_after,norm=colors.LogNorm(vmin = 0.01, vmax = 0.1))
im =ax[0].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),sli_morn,vmin = vmin, vmax = vmax,cmap=cmap)
im =ax[1].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),sli_mid,vmin = vmin, vmax = vmax,cmap=cmap)
im = ax[2].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),sli_after,vmin = vmin, vmax = vmax,cmap=cmap)
#cb = plt.colorbar(cs)
ax[0].set_title('(a) Summer 7 - 10 AM'); ax[1].set_title('(b) Summer 11 AM - 2 PM'); ax[2].set_title('(c) Summer 4 - 7 PM');

im = ax[3].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),wsli_morn,vmin = vmin, vmax = vmax,cmap=cmap)
im = ax[4].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),wsli_mid,vmin = vmin, vmax = vmax,cmap=cmap)
im = ax[5].pcolormesh(lon[xl].T[yl:yu+3],np.arange(0,35),wsli_after,vmin = vmin, vmax = vmax,cmap=cmap)
ax[3].set_title('(d) Winter 7 - 10 AM'); ax[4].set_title('(e) Winter 11 AM - 2 PM'); ax[5].set_title('(f) Winter 4 - 7 PM');

# make lake line
[ax[i].axvline(x=-87.6,alpha=0.8,c = 'k', linestyle="dotted") for i in range(len(ax))] # line showing lakeshore
[ax[i].set_ylim(0,25) for i in range(len(ax))] # line showing lakeshore
fig.colorbar(im, ax=ax.tolist())
#plt.show()

plt.savefig('no2_profile_d02.png',transparent=True)








# make gif
for day in range(7):
	sli= [np.array([np.array(base[day]['O3'][t][l][xl:xu]).T[yl:yu].mean(axis=-1) for l in range(35)]) for t in range(24)]
	for t in range(len(sli)):
		plt.figure()
		plt.pcolormesh(lon[xl].T[yl:yu],np.arange(0,len(sli[0])),sli[t],vmin=0,vmax=.08)
		plt.title('Day %i, Hour %i'%(day,t))
		plt.xlabel('Longitude')
		plt.ylabel('Layer')
		plt.savefig('Chi_o3_day_%i_hour_%i.png'%(day,t))
		plt.close()


#
gif_name = 'o3_column'
fps = 6
file_list = ['Chi_o3_day_%i_hour_%i.png'%(day,t) for day in range(7) for t in range(24)]
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)



