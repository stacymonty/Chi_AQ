#!/usr/bin/env python

# plot_cmaq_may2021.py

# ---------------------------------------------------------------------
# Stacy Montgomery, NOV 2018 - DEC 2018
# This program takes the cropped l2 files and regrids the data to new domain.
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
# ---------------------------------------------------------------------

# dir to grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 

dir_epa='/projects/b1045/montgomery/CMAQcheck/'

dir_cmaq_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'
dir_cmaq_d03_wint='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'

#names of lat lons in the cmaq grid
la,lo='lat','lon' # for 1.3km
la,lo='LAT','LON' # for 4km

# CMAQ RUN things
domain=['d03']*3
time='hourly'
year='2018'
month='8'
ssn = 'Summer'

var = ['NO2','O3','PM25_TOT']*2
var_tit=[r'NO$_2$',r'O$_3$',r'PM$_{2.5,TOT}$']
#epa_files =[dir_epa+'%s_%s_%s.csv'%(time,epa_code[i],year,) for i in range(len(epa_code))]
epa_files =[dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],domain[i],year,month,) for i in range(len(domain))]

year='2019'
month='1'
ssn = 'Winter'

ep2 = [dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],domain[i],year,month,) for i in range(len(domain))]

epa_files = epa_files +ep2

startswith = 'COMBINE_ACONC'

# ---------------------------------------------------------------------

def pull_cmaq(dir_CMAQ,startswith,cmaq_var):
   #pull model files from given directoy
   onlyfiles = next(os.walk(dir_CMAQ))[2]
   onlyfiles.sort() # so that searching for dates are easier
   # pull only CONC files
   fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
   # get data files
   ncfile_CMAQ_base = [Dataset(dir_CMAQ+ fnames_CMAQ[i],'r') for i in range(len(fnames_CMAQ))]
   units_cmaq = [ncfile_CMAQ_base[0][cmaq_var[i]].units for i in range(len(cmaq_var))]
   return ncfile_CMAQ_base, units_cmaq



def get_min_max_cmaq(base,var,hrs, mask=False, ma = np.zeros(3)):
	basel=[base[i][var][hr] for i in range(len(base)) for hr in hrs]
	basel=np.array(basel)
	if mask==True: 
		base_max = np.array([basel[i][0][~mask].max() for i in range(len(basel))])
		base_min = np.array([basel[i][0][~mask].min() for i in range(len(basel))])
		base_mean = np.array([basel[i][0][~mask].mean() for i in range(len(basel))])
	else:
		base_max = np.array([basel[i][0].max() for i in range(len(basel))])
		base_min = np.array([basel[i][0].min() for i in range(len(basel))])
		base_mean = np.array([basel[i][0].mean() for i in range(len(basel))])
	return base_max,base_min,base_mean


def get_avg_cmaq(base,var,hrs, mask=False, ma = np.zeros(3)):
	basel=[base[i][var][hr] for i in range(len(base)) for hr in hrs]
	basel=np.array(basel)
	return np.mean(basel,axis=0)[0]

def get_avg_epa(epa_file):
#for t in range(1):
	ef = epa_file
	epa = pd.read_csv(ef)
	epa_drop = pd.DataFrame([epa.level_0.tolist(),epa['Sample Measurement'].tolist(),epa['CMAQ'].tolist(),epa['Latitude'].tolist(),epa['Longitude'].tolist()]).T
	epa_drop.columns = ['Datetime','Sample Measurement','CMAQ','Lat','Lon']
	epa_drop['Sample Measurement'] = epa_drop['Sample Measurement'].astype(float)
	return epa_drop.groupby(['Lat','Lon']).mean().reset_index()


def get_min_max_epa(epa_file):
#for t in range(1):
	ef = epa_file
	epa = pd.read_csv(ef)
	epa_drop = pd.DataFrame([epa.level_0.tolist(),epa['Sample Measurement'].tolist(),epa['CMAQ'].tolist()]).T
	epa_drop.columns = ['Datetime','Sample Measurement','CMAQ']
	epa_drop.Datetime = pd.to_datetime(epa_drop.Datetime)
	epa_drop = epa_drop.set_index('Datetime')
	#
	fmax_epa,fmin_epa,fmean_epa,fmax_cmaq,fmin_cmaq,fmean_cmaq = [],[],[],[],[],[]
	for i in range(744):
		f=epa_drop.loc[epa_drop.index[i]]
		fmax_epa.append(f.max().tolist()[0])
		fmin_epa.append(f.min().tolist()[0])
		fmean_epa.append(f.mean().tolist()[0])
		fmax_cmaq.append(f.max().tolist()[1])
		fmin_cmaq.append(f.min().tolist()[1])
		fmean_cmaq.append(f.mean().tolist()[1])
	# Plot by max/min/avg
	return epa_drop.index[0:744],fmax_epa,fmin_epa,fmean_epa,fmax_cmaq,fmin_cmaq,fmean_cmaq


# ---------------------------------------------------------------------
#START CODE
# ################### ################### ################### ##################


base,t_u = pull_cmaq(dir_cmaq_d03,startswith,var[0:3])
base_wint,t_u = pull_cmaq(dir_cmaq_d03_wint,startswith,var[0:3])

datas = [get_avg_cmaq(base,var[i],hrs) for i in range(len(var[0:3]))] + [get_avg_cmaq(base_wint,var[i],hrs) for i in range(len(var[0:3]))] 


# adjustable plotting parts
llx=Dataset(dir+ll,'r')
lat,lon=llx['lat'][:],llx['lon'][:]


#START PLOT
# ################### ################### ################### ##################

c = ['Orchid','Blue','limegreen']
c2 = ['Purple','Navy','darkgreen']

hrs = np.arange(0,24)


units = [r'ppb',r'ppb',r'ug/m$^3$']*3
vmins,vmaxs = [0,25,5,0,25,5],[20,45,13,20,45,13]

titles = [r'Summer NO$_2$ ',r'Winter NO$_2$',
r'Summer O$_3$ ',r'Winter O$_3$',
r'Summer PM$_2.5$ ',r'Winter PM$_2.5$']


cmaps = ['Purples','Blues','Greens']*2


figtit = 'monthly_average_with_overlay.png'
#--- fig

def create_fig(lon,lat,base,datas,varS,vmins,vmaxs,cmaps,units, titles,figtit,show=False,save=False):
#
	crs_new = ccrs.PlateCarree()
	fig, axs = plt.subplots(nrows=3,ncols=2,subplot_kw={'projection': crs_new},figsize=(8,7))
	axs = axs.T.ravel()
	axs[0].set_ylabel(r'NO$_2$')
	axs[1].set_ylabel(r'O$_3$')
	axs[3].set_ylabel(r'PM$_2.5$')
	axs[0].set_title('Summer')
	axs[3].set_title('Winter')
	#
	for i in range(len(axs)):
		print(varS[i])
		epa = get_avg_epa(epa_files[i])
		#if i < 3: data = get_avg_cmaq(base,varS[i],hrs)
		#else: data = get_avg_cmaq(base_wint,varS[i],hrs)
		data = datas[i]
		vmin = vmins[i]
		vmax = vmaxs[i]
		title = titles[i]
		cmap = cmaps[i]
		if varS[i] == 'O3': epa['Sample Measurement'] = epa['Sample Measurement']*1000+10
		levels =  list(np.arange(vmin,vmax,(vmax-vmin)/10))+[vmax]
		#plot
		cs=axs[i].pcolormesh(lon,lat, data,transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)#
		cs2 = axs[i].scatter(epa.Lon,epa.Lat,c=epa['Sample Measurement'],cmap = cmap, vmin = vmin, vmax = vmax,s=40,edgecolors = 'black')
		# add limits
		#
		x=[lon.min(),lon.max()]
		y=[lat.min(),lat.max()]
		axs[i].set_extent([x[0]+.5,x[1]-.5,y[0]+.5,y[1]-.5],crs= crs_new)
		#
		#if i ==3 or i ==4 or i == 5:
		if i < 100:
			cbar=plt.colorbar(cs,boundaries= levels,fraction=0.028, pad=0.02,ax=axs[i])
			#
			cbar.set_ticks(levels)
			cbar.set_label(units[i])
		# add features
		states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',edgecolor='black',facecolor='none',scale='10m',alpha = 0.3)
		borders = cfeature.NaturalEarthFeature(scale='50m',category='cultural',name='admin_0_countries',edgecolor='black',facecolor='none',alpha=0.6)
		land = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='none')
		axs[i].add_feature(land, edgecolor='black')
		axs[i].add_feature(borders, edgecolor='black')
		axs[i].add_feature(states_provinces, edgecolor='black')
		#axs[i].set_title(i)
		# add title
		#axs[i].set_title(title)
	plt.tight_layout()
	#
	#
	if save == True: plt.savefig(figtit)
	#
	if show==True: plt.show()


create_fig(lon,lat,base,datas,var,vmins,vmaxs,cmaps,units, titles,figtit,show=True,save=False)



