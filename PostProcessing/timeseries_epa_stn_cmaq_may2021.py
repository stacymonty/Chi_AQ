#!/usr/bin/env python

# Time series
# Stacy Montgomery
# May 2021

#---------------------------------------------------------#
from datetime import timedelta, date,datetime; 
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from wrf import latlon_coords, getvar
import glob, os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Point, shape, Polygon
import fiona
from shapely.ops import unary_union, cascaded_union
from geopandas.tools import sjoin
import geopandas as gpd; import geoplot; 
import glob; 
import os;
#---------------------------------------------------------#

dir_epa='/projects/b1045/montgomery/CMAQcheck/'

dir_cmaq_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'
dir_cmaq_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/postprocess/'

#names of lat lons in the cmaq grid
la,lo='lat','lon' # for 1.3km
la,lo='LAT','LON' # for 4km

# CMAQ RUN things
domain=['d03']*3+['d02']*3
time='hourly'
year='2018'
month='8'
ssn = 'Summer'

var = ['NO2','O3','PM25_TOT']*2
var_tit=[r'NO$_2$',r'O$_3$',r'PM$_{2.5,TOT}$']
#epa_files =[dir_epa+'%s_%s_%s.csv'%(time,epa_code[i],year,) for i in range(len(epa_code))]
epa_files =[dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],domain[i],year,month,) for i in range(len(domain))]

#------ DATERANGE

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



def mask_given_shapefile(lon,lat,shapefile):
   '''
   Make a mask given a shapefile
   lon - array of grid lons
   lat - array of grid lats
   shapefile - geopandas geodataframe shapefile
   '''
   union=gpd.GeoSeries(unary_union(shapefile.geometry))
   mask=np.ones(lon.shape,dtype=bool)
   mask[:] = True
   for i in range(len(lon)):
       for j in range(len(lon[0])):
          pt = Point(lon[i][j],lat[i][j])
          if pt.within(union[0]):
             mask[i][j] = False
   #
   return mask



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


def get_min_max_epa(epa_file):
#for t in range(1):
	ef = epa_files[0]
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



#epa_drop.groupby('Datetime').mean()




#START CODE
# ################### ################### ################### ##################

c = ['Orchid','Blue','limegreen']
c2 = ['Purple','Navy','darkgreen']

fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(7.5,9))
axs=axs.ravel()

for i in range(3):
	ax = axs[i]
	dt,fmax_epa,fmin_epa,fmean_epa,fmax_cmaq,fmin_cmaq,fmean_cmaq = get_min_max_epa(epa_files[i])
	#station
	#if var[i]=='O3': fmax_epa,fmin_epa,fmean_epa = np.array(fmax_epa)*1000,np.array(fmin_epa)*1000,np.array(fmean_epa)*1000
	ax.plot(dt, fmean_epa, '--',color=c[i],label='Station Mean')
	ax.fill_between(dt,fmin_epa, fmax_epa,facecolor=c[i],alpha=0.1)
	#cmaq
	ax.plot(dt[0:744], fmean_cmaq[0:744],'--',color=c2[i],label='CMAQ Mean')
	ax.fill_between(dt[0:744],fmin_cmaq[0:744], fmax_cmaq[0:744],facecolor=c2[i],alpha=0.1)
	#extra info
	if var[i]== 'O3': ax.set_ylim([0,100])
	else: ax.set_ylim([0,50])
	ax.set_xlim(dt[0],dt[-1])
	# set week major ticks
	fmt_wk = mdates.DayLocator(interval=7)
	ax.xaxis.set_major_locator(fmt_wk)
	# set dayminor ticks
	fmt_day = mdates.DayLocator()
	ax.xaxis.set_minor_locator(fmt_day)
	# format title
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	if var[i] == 'PM25_TOT': ax.set_ylabel(var[i]+r' (ug/m$^3$)')
	else: ax.set_ylabel(var[i]+' (ppb)')
	ax.legend()
	#ax.set_title(var_tit[i])
	if i ==0: ax.set_title(ssn)

plt.tight_layout()

plt.savefig('timseries_epa_cmaq_%s-%s.png'%(year,month))

plt.show()


# CMAQ
# ################### ################### ##################


#pull model files from given directoy
onlyfiles = next(os.walk(dir_cmaq_d03))[2]
onlyfiles.sort() # so that searching for dates are easier
startswith = 'COMBINE_ACONC'

# pull only CONC files
fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
fnames_CMAQ = fnames_CMAQ[:-1]

#get lat lon from grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc'
llx=Dataset(dir+ll,'r')
lat,lon=llx['lat'][:],llx['lon'][:]

# shapes and directories == https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=State+Legislative+Districts
path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
path2 ='/projects/b1045/montgomery/shapefiles/Chicago/cook/Cook_County_Border.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path2)
mask = mask_given_shapefile(lon,lat,chi_shapefile)

#pull temp
#base,t_u = pull_cmaq(dir_cmaq_d03,startswith,var[0:3])
#base_max,base_min,base_mean = get_min_max_cmaq(base,var[0],hrs)
#base_max_chi,base_min_chi,base_mean_chi = get_min_max_cmaq(base,var[0],hrs,mask=True,ma = mask)


base,t_u = pull_cmaq(dir_cmaq_d03,startswith,var[0:3])
hrs = np.arange(0,24)
dt,fmax_epa,fmin_epa,fmean_epa,fmax_cmaq,fmin_cmaq,fmean_cmaq = get_min_max_epa(epa_files[0])

c = ['Orchid','Blue','limegreen']
c2 = ['Purple','Navy','darkgreen']

fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(7.5,9))
axs=axs.ravel()

for i in range(3):
	ax = axs[i]
	base_max,base_min,base_mean = get_min_max_cmaq(base,var[i],hrs)
	base_max_chi,base_min_chi,base_mean_chi = get_min_max_cmaq(base,var[i],hrs,mask=True,ma = mask)
	#dt,fmax_epa,fmin_epa,fmean_epa,fmax_cmaq,fmin_cmaq,fmean_cmaq = get_min_max_epa(epa_files[i])
	#
	ax.plot(dt, base_mean[0:744], '--',color=c[i],label='Domain Mean')
	ax.fill_between(dt,base_min[0:744], base_max[0:744],facecolor=c[i],alpha=0.1)
	#cmaq
	ax.plot(dt[0:744], base_mean_chi[0:744],'--',color=c2[i],label='Chicago Mean')
	ax.fill_between(dt[0:744],base_min_chi[0:744], base_max_chi[0:744],facecolor=c2[i],alpha=0.1)
	#extra info
	if var[i]== 'O3': ax.set_ylim([0,100])
	else: ax.set_ylim([0,50])
	ax.set_xlim(dt[0],dt[-1])
	# set week major ticks
	fmt_wk = mdates.DayLocator(interval=7)
	ax.xaxis.set_major_locator(fmt_wk)
	# set dayminor ticks
	fmt_day = mdates.DayLocator()
	ax.xaxis.set_minor_locator(fmt_day)
	# format title
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	if var[i] == 'PM25_TOT': ax.set_ylabel(var[i]+r' (ug/m$^3$)')
	else: ax.set_ylabel(var[i]+' (ppb)')
	ax.legend()
	#ax.set_title(var_tit[i])
	if i ==0: ax.set_title(ssn)

plt.tight_layout()

plt.savefig('timseries_ONLY_cmaq_%s-%s.png'%(year,month))

plt.show()














#locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
#formatter = mdates.ConciseDateFormatter(locator)
#ax.xaxis.set_major_locator(locator)
#ax.xaxis.set_major_formatter(formatter)




	#PLOT BY STATIONS
	#epa['latlon'] = [(epa.Longitude.tolist()[i],epa.Latitude.tolist()[i]) for i in range(len(epa.Latitude))]
	#lalo = epa.Latitude.unique().tolist(),epa.Longitude.unique().tolist()
	#epa_drop = epa.dropna(axis=0,subset=['Latitude'])
	#epa_drop_lalo = epa_drop.Latitude.unique()
	#fig,ax = plt.subplots()
	#
	#for i in epa_drop.Latitude.unique():
#		tmp = epa_drop[epa_drop['Latitude']==i]
#		ax.scatter(tmp['level_0'],tmp['Sample Measurement'],label = i)









