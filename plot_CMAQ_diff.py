#!/bin/python

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

# USER INPUT

# shapes and directories == https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=State+Legislative+Districts
path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)

# dir to grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 

# dir to model files
dir_CMAQ = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_Amy_noBUS_1.33km_sf_rrtmg_5_8_1_v3852/'


dir_CMAQ_BASE = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/' # experimental choice

dir_WRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'

dir_EPA = '/home/asm0384/ChicagoStudy/inputs/EPA_hourly_station_data/'

#write out monthly files to csv?
# read in monthly files from csv?
writeoutcsv = True
writeincsv = False
show = True

#variables of interest
cmaq_var=['O3','NO2','NO','NOX','CO','ISOP','SO2','FORM','PM25_TOT','PM10']
cmaq_var=['O3','NO2','NO','CO','ISOP','SO2','FORM']

startswith = "CCTM_ACONC_"

# ---------------------------------------------------------------------
# USER DEF FUNCTIONS
# ---------------------------------------------------------------------

#pull in cmaq
startswith = "CCTM_ACONC"
hours = range(0,12) # consider offset range(23,4)

#writeoutcsv = false
def pull_CMAQ(dir_CMAQ_BASE,startswith,cmaq_var,version):
   #pull files from given directoy
   onlyfiles = next(os.walk(dir_CMAQ_BASE))[2]
   onlyfiles.sort() # so that searching for dates are easier
   fnames_CMAQ_BASE = [x for x in onlyfiles if x.startswith(startswith)]
   numfiles=(len(fnames_CMAQ))
   ncfile_CMAQ_base = [Dataset(dir_CMAQ_BASE+ fnames_CMAQ_BASE[i],'r') for i in range(len(fnames_CMAQ_BASE))]
   units_cmaq = [ncfile_CMAQ_base[0][cmaq_var[i]].units for i in range(len(cmaq_var))]
   #full day conc
   cmaq_avgs_BASE = []
   cmaq_avgs_daily_BASE  = []
   cmaq_avgs_hourly_BASE  = []
   # make averages for cmaq base
   for i in range(len(cmaq_var)):
      tmp = np.asarray([ncfile_CMAQ_base[j][cmaq_var[i]] for j in range(len(ncfile_CMAQ_base))])
      # tmp = np.asarray([ncfile_CMAQ_base[j][cmaq_var[i]][hours] for hours in range(hours) for j in range(len(ncfile_CMAQ_base))])
      hourly = np.average(tmp,axis=0) # hour by hour concs
      daily = np.average(tmp,axis=1) # daily average concs
   #
      monthly = np.average(daily,axis=0)
      if writeoutcsv == True: pd.DataFrame(monthly[0]).to_csv(cmaq_var[i]+'_'+version+'_BASE_2018_aug.csv', header=False,index=False) 
      cmaq_avgs_BASE.append(monthly[0])
      cmaq_avgs_daily_BASE.append(daily)
      cmaq_avgs_hourly_BASE.append(hourly)
      #return
      print('Done with ' +cmaq_var[i])
#return
   return cmaq_avgs_BASE, cmaq_avgs_daily_BASE, cmaq_avgs_hourly_BASE, units_cmaq


#writeoutcsv = false



#plot cmaq

#plotting loop
def plot_cmaq(monthly_tot,var_tot,title_2,cmap,vmaxs,vmins,crs_new,show,add_epa,version,div,shaped):
   for i in range(0,len(monthly_tot)):
   #for i in range(1):
      # set var for plot
      var= var_tot[i]
      data= np.asarray(monthly_tot[i])*1000
      if var == 'RAINC': pass
      else:
         if i<len(cmaq_var): plon,plat = lon,lat
         else: plon,plat = wlon,wlat
         # make fig object
         fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 6))
         #set up data for plotting via levels
         vmax,vmin=vmaxs[i],vmins[i]
         levels = np.arange(vmin, vmax, (vmax-vmin)/10)
         # set boundary as outer extent by making a matplotlib path object and adding that geometry
         if shaped: axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
         axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
         #plot the gridded data by using contourf
         if div==False: cs=plt.pcolormesh(plon,plat,data,cmap= cmap , transform=crs_new, vmin=vmin,vmax=vmax)
         if div == True:
            divnorm = colors.DivergingNorm(vmin=vmin, vcenter=0, vmax= vmax)
            cs=plt.pcolormesh(plon,plat,data,cmap= cmap , transform=crs_new, vmin=vmin,vmax=vmax, norm=divnorm)
#for i in range(1):
         # add landmarks with scatterplot
         midway=  -87.7522,41.7868
         ohare = -87.9073, 41.9842
         loop =  -87.6251,41.8786
         axs.annotate(xy=midway,s="*",color='white')
         axs.annotate(xy=ohare,s="*",color='white')
         axs.annotate(xy=loop,s="*",color='white')
         # set axes extents from shapefile
         x=[chi_shapefile.bounds.minx.min(), chi_shapefile.bounds.maxx.max()] 
         y=[chi_shapefile.bounds.miny.min(), chi_shapefile.bounds.maxy.max()]
         #
         if shaped: axs.set_extent([x[0],x[1],y[0],y[1]],crs= crs_new)
         else: axs.set_extent([x[0]-.3,x[1]+.3,y[0]-.3,y[1]+.3],crs= crs_new)
         # title
         axs.set_title(var+title_2)
         #add colorbar and label
         cbar=plt.colorbar(cs,boundaries=levels)
         cbar.set_ticks(levels)
         # add state lines
         states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
         #
         land = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                                        edgecolor='black',
                                        facecolor='none')
         axs.add_feature(land, edgecolor='black')
         axs.add_feature(states_provinces, edgecolor='black')
         #add epa monitors if its CMAQ
         if add_epa == True:
            if i < len(cmaq_var):
               try: axs.scatter(epa_avgs_latlon[i]['Longitude'],epa_avgs_latlon[i]['Latitude'],c=epa_avgs_latlon[i]['Arithmetic Mean'],vmin = vmin, vmax= vmax,s=75, cmap = cmap,edgecolors = 'black')
               except: pass
         #savefig
         plt.savefig(var+version+'v7.png',transparent=True)
         if show == True: plt.show()







# ---------------------------------------------------------------------
# START
# ---------------------------------------------------------------------

#pull files from given directoy
onlyfiles = next(os.walk(dir_CMAQ))[2]
onlyfiles.sort() # so that searching for dates are easier

# pull only CONC files
fnames_CMAQ = [x for x in onlyfiles if x.startswith(startswith)]
numfiles=(len(fnames_CMAQ))

#dates
dates=[fnames_CMAQ[i].split(startswith)[1].split('_')[1].split('.nc')[0] for i in range(len(fnames_CMAQ))]

#get lat lon from grid file
llx=Dataset(dir+ll,'r')
lat,lon=llx['lat'][:],llx['lon'][:]


#-------------------- EPA -------------------------------

param_codes = pd.read_csv(dir_EPA+'parameters.csv')
#epa codes for variables
epa_codes = []
for i in range(len(cmaq_var)):
   try:
      code = int(param_codes[param_codes['Parameter Abbreviation']== cmaq_var[i]]['Parameter Code'])
      epa_codes.append(code)
   except:
      if cmaq_var[i] == 'PM25_TOT': epa_codes.append(81104)
      elif cmaq_var[i] == 'ISOP': epa_codes.append(43243)
      else: epa_codes.append(np.nan)

# make monthly averages of EPA stations from daily data

units_epa = []

for i in range(len(cmaq_var)):
    try: 
       tmp = pd.read_csv(dir_EPA+'daily_'+str(epa_codes[i])+'_2018.csv')
       units_epa.append(tmp['Units of Measure'].unique()[0])
    except:
       units_epa.append(np.nan)

# make averages for epa
epa_avgs_latlon = []
for i in range(len(cmaq_var)):
    try: 
       tmp = pd.read_csv(dir_EPA+'daily_'+str(epa_codes[i])+'_2018.csv')
       tmp = tmp[tmp['County Name']=='Cook']
       tmp['datetime'] = pd.to_datetime(tmp['Date Local'])
       mask = (tmp['datetime'] >= pd.to_datetime(dates)[0]) & (tmp['datetime'] <= pd.to_datetime(dates)[-1])
       tmp = tmp.loc[mask]
       epa_lat,epa_lon= tmp['Latitude'].unique(), tmp['Longitude'].unique()
       tmp_avg = tmp.groupby(['Longitude','Latitude','datetime'])['Arithmetic Mean'].mean().reset_index()
       tmp_avg.to_csv(dir_EPA + cmaq_var[i]+ '_'+ version+ '_daily_avg_by_ChiMonitor_Aug2018.csv')
       tmp_avg = tmp.groupby(['Longitude','Latitude'])['Arithmetic Mean'].mean().reset_index()
       tmp_avg.to_csv(dir_EPA + cmaq_var[i]+ '_' + version +'_by_ChiMonitor_Aug2018.csv')
       epa_avgs_latlon.append(tmp_avg)
    except: 
       print('No EPA file for ' + cmaq_var[i])
       epa_avgs_latlon.append(np.nan)
       #epa_avgs.append(np.nan)

# ppm to ppb
epa_avgs_latlon[4]['Arithmetic Mean'] = epa_avgs_latlon[4]['Arithmetic Mean']*1000
epa_avgs_latlon[0]['Arithmetic Mean'] = epa_avgs_latlon[0]['Arithmetic Mean']*1000


#----------  PULL IN CMAQ

#pull in cmaq

startswith = "CCTM_ACONC"
version = '_aug2018_monthly_nobusdiff'
base_monthly, base_daily, base_hourly, units = pull_CMAQ(dir_CMAQ_BASE,startswith,cmaq_var,version)
nobus_monthly, nobus_daily, nobus_hourly, units = pull_CMAQ(dir_CMAQ,startswith,cmaq_var,version)


#----------  START PLOTTING
import cartopy.feature as cfeature 

# projection
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])


# version and title STUFF
title_2 = " nobus-base, Aug. 2018"
var_tot = cmaq_var 
monthly_tot = [nobus_monthly[i]-base_monthly[i] for i in range(len(base_monthly))]
cmap = 'magma_r'
version = '_nobusDIFF_aug2018_'


# SET RANGES
vmaxs,vmins = [round(np.percentile(monthly_tot[i],99.99),5)*1000 for i in range(len(monthly_tot))],[round(np.percentile(monthly_tot[i],0.1),5)*1000 for i in range(len(monthly_tot))]

# DO WE EPA SCATTER
add_epa = False #True

#is it a difference map
div = True
if div==True: cmap = 'RdBu_r'

shaped= True

plot_cmaq(monthly_tot,var_tot,title_2,cmap,vmaxs,vmins,crs_new,show,add_epa,version,div,shaped)


#---- base case
monthly_tot = base_monthly #[nbase_monthly[i] for i in range(len(base_monthly))]
version = '_SPOT_'
title_2 = " , Aug. 2018"
vmaxs,vmins = [round(np.percentile(monthly_tot[i],99.99),5)*1000 for i in range(len(monthly_tot))],[round(np.percentile(monthly_tot[i],98),5)*1000 for i in range(len(monthly_tot))]
vmaxs[0],vmins[0] =40,30
shaped = False
add_epa = True
plot_cmaq(monthly_tot,var_tot,title_2,cmap,vmaxs,vmins,crs_new,show,add_epa,version,div,shaped)

