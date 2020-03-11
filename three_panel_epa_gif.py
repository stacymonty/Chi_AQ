#!/usr/bin/env python3

#libraries
from matplotlib import pyplot as plt ; from matplotlib import colors
import numpy as np; import numpy.ma as ma; from matplotlib.patches import Path, PathPatch
import pandas as pd; from shapely.geometry import Point, shape, Polygon;import fiona
from shapely.ops import unary_union, cascaded_union; from geopandas.tools import sjoin
import geopandas as gpd; import glob; import os; from datetime import timedelta, date;
from netCDF4 import Dataset; from cartopy import crs as ccrs; from cartopy.io.shapereader import Reader
import matplotlib.path as mpath; import seaborn as sns; import timeit; from cartopy import crs as ccrs

import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from cartopy.feature import NaturalEarthFeature as cfeature


dir_EPA = '/home/asm0384/ChicagoStudy/inputs/EPA_hourly_station_data/'


#----------------------------------------------------------------------------------------
# User input
#----------------------------------------------------------------------------------------

gmt_offset = 7

# directory to model files
dir_CMAQ='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'
dir_SMOKE='/projects/b1045/jschnell/ForAmy/smoke_out/base/'
dir_WRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'

#directory to grid file
dir_GRID='/projects/b1045/jschnell/ForStacy/latlon_ChicagoLADCO_d03.nc' 

# dir to lat lon
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 

# CMAQ RUN things
domain='d03'
time='hourly'
year='2018'
month='8'

#directory to chicago shapefile
dir_shapefile='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'

# this will use just the epa var
cmaq_var=['O3','NO2','NO','CO','ISOP','SO2','FORM','PM25_TOT']
smoke_var=['NO2','NO','CO','ISOP','SO2','FORM']
epa_code=['42401','42602','44201','42101']; var=['SO2','NO2','O3','CO']


# pull epa
dir_epa='/home/asm0384/CMAQcheck/'

epa_condense=[dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[loop],domain,year,month) for loop in range(len(epa_code))]
so2_epa,no2_epa,o3_epa,co_epa = [pd.read_csv(epa_condense[i]) for i in range(len(epa_condense))]

# set up shape of cmaq indexing
shape = (32,24,1,288,315)

#-------------------------------------------------------------------------------------------
# User defined functions
#-------------------------------------------------------------------------------------------

def pull_CMAQ(dir_CMAQ_BASE,startswith,cmaq_var,version):
   #pull files from given directoy
#for i in range(1):
   onlyfiles = next(os.walk(dir_CMAQ_BASE))[2]
   onlyfiles.sort() # so that searching for dates are easier
   fnames_CMAQ_BASE = [x for x in onlyfiles if x.startswith(startswith)]
   ncfile_CMAQ_base = [Dataset(dir_CMAQ_BASE+ fnames_CMAQ_BASE[i],'r') for i in range(len(fnames_CMAQ_BASE))]
   units_cmaq = [ncfile_CMAQ_base[0][cmaq_var[i]].units for i in range(len(cmaq_var))]
   #full day conc
   cmaq_avgs_BASE = []; cmaq_avgs_daily_BASE  = []; cmaq_avgs_hourly_BASE  = []; all_hours =[]
   # make averages for cmaq base
   for i in range(len(cmaq_var)):
   #for i in range(1):
      tmp = np.asarray([ncfile_CMAQ_base[j][cmaq_var[i]] for j in range(len(ncfile_CMAQ_base))])
      hourly = np.average(tmp,axis=0) # hour by hour concs
      daily = np.average(tmp,axis=1) # daily average concs
   #
      monthly = np.average(daily,axis=0)
      #if writeoutcsv == True: pd.DataFrame(monthly[0]).to_csv(cmaq_var[i]+'_'+version+'_2018_aug.csv', header=False,index=False) 
      cmaq_avgs_BASE.append(monthly[0])
      cmaq_avgs_daily_BASE.append(daily)
      cmaq_avgs_hourly_BASE.append(hourly)
      all_hours.append(tmp)
      #return
      if Chatty: print('Done with ' +cmaq_var[i])
#return
   return cmaq_avgs_BASE, cmaq_avgs_daily_BASE, cmaq_avgs_hourly_BASE, all_hours, units_cmaq


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def add_gmt_offset(list_of_hours,gmt_offset):
    update_list = []
    for i in range(len(list_of_hours)):
        if list_of_hours[i] + gmt_offset > 23:
           update_list.append(list_of_hours[i] + gmt_offset - 24)
        elif list_of_hours[i] + gmt_offset < 0:
           update_list.append(list_of_hours[i] + gmt_offset + 24)
        else: update_list.append(list_of_hours[i] + gmt_offset)
    # return
    return update_list


def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
 # stn -- points in a list (list, can be a list of just 1) 
 # wrf -- gridded wrf lat/lon (np.array)
 #for iz in range(1):
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
    #
    xx=[xx[i][0] for i in range(len(xx))];yy=[yy[i][0] for i in range(len(yy))]
    #return indices list
    return xx, yy



#-------------------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------------------


# get dates
startswith = 'COMBINE_ACONC_'
onlyfiles = next(os.walk(dir_CMAQ))[2]
onlyfiles.sort() # so that searching for dates are easier
fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
dates=[fnames_CMAQ[i].split(startswith)[1].split('.nc')[0] for i in range(len(fnames_CMAQ))]

# get lat lon
llx=Dataset(dir+ll,'r')
lat,lon=llx['lat'][:],llx['lon'][:]


# ============================================
# make fancy plot to plot full days
# ============================================
'''
no2_epa = chemical data over domain with nearest cmaq pixel. dataframe.

'''

def tri_plot(epa, ncfile_CMAQ, var, v, picdir, printout=False):
#data prep
   epa['level_0']=pd.to_datetime(epa['level_0'])
   epa['month-day'] = pd.to_datetime(epa['level_0']).dt.to_period('D')
   #epa=epa.groupby('month-day').mean()
   vmin=round(np.percentile(ncfile_CMAQ[0][var[v]][0][0].ravel(),0.01))
   vmax=round(np.percentile(ncfile_CMAQ[0][var[v]][0][0].ravel(),99.99))
# start plotting
   cmap = 'magma_r'
   crs_new = ccrs.PlateCarree()
   for d in range(shape[0]):
      for h in range(shape[1]):
   # set up fig
         fig = plt.figure(figsize=(10,8))
         #fig.execute_constrained_layout()
         widths = [2, 2]
         heights = [5, 2]
         gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,height_ratios=heights)
   # set up plot
         #
         # PLOT 1
         # make map plot on top
         tmp = epa[epa['level_0']==epa['level_0'][h+d*24]]
         levels = np.arange(vmin, vmax, (vmax-vmin)/10)
         ax = fig.add_subplot(gs[0, :],projection= crs_new)
         cs = ax.pcolor(lon,lat, ncfile_CMAQ[d][var[v]][h][0],transform=crs_new,cmap = cmap,vmin=vmin,vmax=vmax)
         ax.scatter(tmp['Longitude'],tmp['Latitude'],c= tmp['Sample Measurement'], cmap = cmap, vmin = vmin, vmax = vmax,s=75,edgecolors = 'black')
         cbar=plt.colorbar(cs,boundaries=levels,shrink = 0.75,label='ppbV')
         cbar.set_ticks(levels)
         states_provinces = cfeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
         land = cfeature('physical', 'lakes', '10m',edgecolor='black',facecolor='none')
         ax.add_feature(land, edgecolor='black')
         ax.add_feature(states_provinces, edgecolor='black',alpha = 0.5)
         b = .8
         xl,xu,yl,yu = lon.min()+b,lon.max()-b,lat.min()+b,lat.max()-b
         ax.set_extent([xl,xu,yl,yu], crs= crs_new)
         plt.title(var[v]+' on '+str(epa['level_0'][h+d*24]))
   # PLOT 2
   # make 1:1 plot 
         ax1 = fig.add_subplot(gs[1, 1])
         tmp = epa[epa['level_0']==epa['level_0'][h+d*24]]
   #
         for label in range(len(tmp['County Name'])):
            l = tmp.index[label]
            if tmp['Sample Measurement'][l] == np.nan: print('movin')
            else: ax1.scatter(tmp['Sample Measurement'][l],tmp['CMAQ'][l],label= tmp['County Name'][l],color = plt.get_cmap('Blues',len(tmp))(label))
   #
         plt.xlabel('Sample Measurement (ppbv)'); plt.ylabel('CMAQ (ppbv)')
         #plt.legend( loc='upper center', bbox_to_anchor=(.5, 1.5), ncol=4, prop={'size': 6},)
         scr, mcr = tmp['Sample Measurement'], tmp['CMAQ']
         bad = np.isnan(scr)
         r = round(pearsonr(mcr[~bad],scr[~bad])[0],2)
         plt.title(f'Station vs. CMAQ Pixel: r = {r}')
         plt.xlim([tmp['CMAQ'].min()*.8,tmp['CMAQ'].max()*1.2]); plt.ylim([tmp['CMAQ'].min()*.8,tmp['CMAQ'].max()*1.2])
         ax1.plot([-1000,1000],[-1000,1000],c='black',alpha = 0.75)
   # PLOT 3
   ##make diurnal plot
   #for i in range(1):
         ax2 = fig.add_subplot(gs[1, 0])
         tmp2=epa.groupby('level_0').mean()['Sample Measurement']
         tmp2.plot.line(linestyle='--',color= plt.get_cmap('Blues',8)(1),ax=ax2,label='EPA')
         tmp2=epa.groupby('level_0').mean()['CMAQ']
         tmp2.index.name = 'Dates'
         tmp2.plot.line(color=plt.get_cmap('Blues',8)(5),ax=ax2)
         ax2.scatter(tmp2.index[24*d+h],tmp2[24*d+h], marker='*',color='pink',s=200)
         ax2.set_xlim(tmp2.index[h]+timedelta(days=d-1), tmp2.index[h]+timedelta(days=d+1))
   #
         plt.legend( loc='upper center', ncol=4, prop={'size': 8},)
   #
         plt.savefig(picdir+var[v]+'_'+'day'+str(d)+'_hour'+str(h)+'.png', orientation='landscape')
         plt.close()
         if printout== True: print(f'Done with day {d} hour {h}')


startswith = 'COMBINE_ACONC_'

onlyfiles = next(os.walk(dir_CMAQ))[2]
onlyfiles.sort() # so that searching for dates are easier
fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
ncfile_CMAQ = [Dataset(dir_CMAQ+ fnames_CMAQ[i],'r') for i in range(len(fnames_CMAQ))]


picdir = '/home/asm0384/gifs/'

tri_plot(so2_epa, ncfile_CMAQ, var, 0, picdir, False)


v=1; epa = no2_epa
tri_plot(epa, ncfile_CMAQ, var, v, picdir, False)


v=2; epa = o3_epa
tri_plot(epa, ncfile_CMAQ, var, v, picdir, False)

v=3; epa = co_epa
tri_plot(epa, ncfile_CMAQ, var, v, picdir, False)
