# HELLO!!!!!

# This code is truly nonsensical
# Basically a scratch sheet of coding notes


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
import timeit
from cartopy import crs as ccrs
import cartopy

#------------------------------------------
# User input
#------------------------------------------

gmt_offset = 7

# directory to model files
#dir_CMAQ='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_1.33km_sf_rrtmg_5_10_1_v3852/'
dir_CMAQ='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'
dir_CMAQ = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'


#directory to grid file
dir_GRID='/projects/b1045/jschnell/ForStacy/latlon_ChicagoLADCO_d03.nc' 

#directory to WRF output
#dir_WRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_1.33km_sf_rrtmg_5_10_1_v3852/'
dir_WRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'

#directory to chicago shapefile
dir_shapefile='/home/asm0384/shapefiles/replines/tl_2019_17_sldl.shp'
dir_shapefile='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
lakemichigan='/home/asm0384/shapefiles/lake_michigan/Lake_Michigan_Shoreline.shp' # from https://gis-michigan.opendata.arcgis.com/datasets/5e2911231fe246128d0ff8495935ee85_12/data

# calculations over entire domain or specify bounds?
crop = True

crop_given_shapefile = True # uses max and mins of shapefile

if crop_given_shapefile != True: # then manually set the bounds
    x1,y1,x2,y2,x3,y3,x4,y4= 0,0,0,0,0,0,0,0

#print out statements on time?
Chatty = True


#------------------------------------------
# User defined functions
#------------------------------------------

#'''''''''''''''''''''''''''''''''
def crop_array(x1,y1,x2,y2,x3,y3,x4,y4,lon,lat,gridded_var):
#''''''''''''''''''''''
# x1,y1,x2,y2,x3,y3,x4,y4 = indices of corners of cropping function
# lon,lat,gridded_var = grid of lons, lats, and variable
#''''''''''''''''''''''
#for i in range(1):
    # Make box around chicago to cut data -- specific for satellite, check to make sure the arrays are increasing in size
    # converting lat lon corners to index corners
    #set up zeros array given the bounds
    diffy =max(y1i,y2i,y3i,y4i)[0]-min(y1i,y2i,y3i,y4i)[0]
    diffx=max(x1i,x2i,x3i,x4i)[0]-min(x1i,x2i,x3i,x4i)[0]
    zlon,zlat,z=np.zeros([diffx, diffy]), np.zeros([diffx, diffy]), np.zeros([diffx, diffy])
    # fill out zeros array from the gridded data
    for i in range(diffx):
       for j in range(diffy):
          z[i][j]= gridded_var[min(x1i,x2i,x3i,x4i)[0]+i][min(y1i,y2i,y3i,y4i)[0]+j] 
          zlat[i][j]= lat[min(x1i,x2i,x3i,x4i)[0]+i][min(y1i,y2i,y3i,y4i)[0]+j]
          zlon[i][j]= lon[min(x1i,x2i,x3i,x4i)[0]+i][min(y1i,y2i,y3i,y4i)[0]+j]
    #return values
    return zlon,zlat,z


#'''''''''''''''''''''''''''''''''
def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
#'''''''''''''''''''''''''''''''''
# Find index of points on a gridded array
# stn_lon,stn_lat = list of lat lon points --> lat_list, lon_list = [x1,x2][y1,y2]
# wrf_lon, wrf_lat = np.array of gridded lat lon --> grid_x= np.array([x1,x2,x3],[x4,x5,x6])
# stn -- points in a list (list, can be a list of just 1) 
# wrf -- gridded wrf lat/lon (np.array)
#'''''''''''''''''''''''''''''''''
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


# hours,days,var_crop,var = hour,np.arange(0,len(dates)).tolist(),var_crop,var
#'''''''''''''''''''''''''''''''''
def hourly_average(hours,days,var_crop,var):
#'''''''''''''''''''''''''''''''''
# hours = list of hours. ex: [6,7,8,9]
# days = list of indices, or indices of specific dates you want to pull **MUST BE INDEX NOT DATE
# var = list of variables. ex: ['O3']
#
   avg=[[] for i in range(len(var))]
   for v in range(len(var)):
      tmp =[var_crop[v][day+hour[hr]] for day in days for hr in range(len(hours))]
      avg[v]=np.average(tmp,axis=0)
   #return list avg
   return avg




# Still working on this one
def add_colorbar(cs,vmin,vmax,ax):
   cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
   cbar=plt.colorbar(cs,boundaries=np.arange(vmin,11),ax=ax)
   cbar.set_ticks(np.arange(vmin, vmax, 10))


#
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

#------------------------------------------
# Start data manipulation
#------------------------------------------

# Pull in files
#------------------------------------------

#load chicago shapefile
chi_shapefile  = gpd.GeoDataFrame.from_file(dir_shapefile)

#load in lake michigan and set the projection to the right one
lm= gpd.GeoDataFrame.from_file(lakemichigan)
lm=lm.to_crs({'proj':'longlat', 'ellps':'WGS84', 'datum':'WGS84'})

#get names of files given directoy
onlyfiles = next(os.walk(dir_CMAQ))[2]
onlyfiles=sorted(onlyfiles)
fnames_cmaq = [x for x in onlyfiles if x.startswith("COMBINE_ACONC")]
fnames_wrf= ['wrfout_d01_'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][0:4]+'-'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][4:6]+'-'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][6:]+'_00:00:00' for i in range(len(fnames_cmaq))]

#dates
dates=[fnames_wrf[i].split('wrfout_d01_')[1].split('_')[0] for i in range(len(fnames_wrf))]

#get lat lon from grid file
ll=Dataset(dir_GRID,'r')
lat,lon=ll['lat'][:],ll['lon'][:]

#pull in model files and variables
# for example: finding the difference between the 11th day and the 0th day of NO2:
# cmaq_ncfile[10]['NO2'][0]-cmaq_ncfile[0]['NO2'][0]
cmaq_ncfile= [Dataset(dir_CMAQ+ fnames_cmaq[i],'r') for i in range(len(fnames_cmaq))]
wrf_ncfile=[Dataset(dir_WRF + fnames_wrf[i],'r') for i in range(len(fnames_wrf))]
#emis_ncfile=[Dataset(dir_EMIS + fnames_wrf[i],'r') for i in range(len(fnames_wrf))]

#variables of interest
var=['O3','NO2','NO','NOX','CO','ISOP','SO2','FORM','PM25_TOT','PM10']
wrf_var=['T2','PSFC','RAINC','RAINNC','Q2','V10','U10']
#smoke_var = ['NO2','NO','CO','ISOP','SO2', 'FORM']

var=['NO2','O3','CO']
wrf_var=['T2','RAINC','RAINNC','Q2','V10','U10']

units_cmaq = [cmaq_ncfile[0][var[i]].units for i in range(len(var))]
units_wrf = [wrf_ncfile[0][wrf_var[i]].units for i in range(len(wrf_var))]
#units_smoke = [emis_ncfile[0][smoke_var[i].units for i in range(len(smoke_var))]


#wrflatlon
wrflon, wrflat = wrf_ncfile[0]['XLONG'][0],wrf_ncfile[0]['XLAT'][0]

#------------------------------------------
#------------------------------------------
#               CMAQ
#------------------------------------------
#------------------------------------------

# Crop area
#------------------------------------------

var_crop = [[] for i in range(len(var))]
# --> o3_crop,no2_crop,pm25_crop = var_crop[0],var_crop[1],var_crop[2]
# --> variable at given day/hour = var_crop[variable][day*hour]

var_crop_wrf = [[] for i in range(len(wrf_var))]
# -->'T2','PSFC','RAINC','RAINNC','Q2'= var_crop_wrf[0], var_crop_wrf[1],....
# --> variable at given day/hour = var_crop_wrf[variable][day*hour]

# for outside the shapefile domain, how much larger outside shapefile boundaries for crop
buffer = 0.1

if crop == True:
    if Chatty == True: print("Starting cropping...")
    started=timeit.time.time()
    # find bounding lat lons given chi shapefile
    if crop_given_shapefile == True:
        x1,y1,x2,y2,x3,y3,x4,y4=[min(chi_shapefile.bounds.minx)-buffer],[min(chi_shapefile.bounds.miny)-buffer],[max(chi_shapefile.bounds.maxx)+ buffer],[max(chi_shapefile.bounds.maxy+.05)],[min(chi_shapefile.bounds.minx-buffer)],[max(chi_shapefile.bounds.maxy+ buffer)],[max(chi_shapefile.bounds.maxx)+.05],[min(chi_shapefile.bounds.miny)-buffer]
    # convert from lat lon to indices
    x1i,y1i=find_index([x1],[y1], np.array(lon), np.array (lat))
    x2i,y2i=find_index([x2],[y2],np.array(lon), np.array (lat))
    x3i,y3i=find_index([x3],[y3],np.array(lon), np.array (lat))
    x4i,y4i=find_index([x4],[y4],np.array(lon), np.array (lat))
    #
    # crop CMAQ variables
    for i in range(len(var)):
        # go through datasets and crop away
        for j in range(len(cmaq_ncfile)): #cmaq_ncfile
           for h in range(24):
              # note: this cmaq file requires an additional indexing bc of masked array ... could be different for different files
              lon,lat,gridded_var= lon,lat, np.array(cmaq_ncfile[j][var[i]][h][0])
              zlon,zlat,z = crop_array(x1i,y1i,x2i,y2i,x3i,y3i,x4i,y4i,lon,lat,gridded_var)
              var_crop[i].append(z)
    #for g in range(1):
    if Chatty==True: print("Done CMAQ in "+str(timeit.time.time()-started)[0:10])
    x1i,y1i=find_index([x1],[y1], np.array(wrflon), np.array (wrflat))
    x2i,y2i=find_index([x2],[y2],np.array(wrflon), np.array (wrflat))
    x3i,y3i=find_index([x3],[y3],np.array(wrflon), np.array (wrflat))
    x4i,y4i=find_index([x4],[y4],np.array(wrflon), np.array (wrflat))
    # crop wrf variables
    for i in range(len(wrf_var)):
        # go through datasets and crop away
        for j in range(len(wrf_ncfile)-1):
           for h in range(24):
              lon,lat,gridded_var= wrflon,wrflat, np.array(wrf_ncfile[j][wrf_var[i]][h])
              zlon_wrf,zlat_wrf,z_wrf = crop_array(x1i,y1i,x2i,y2i,x3i,y3i,x4i,y4i,lon,lat,gridded_var)
              var_crop_wrf[i].append(z_wrf)
 

if Chatty==True: print(str(timeit.time.time()-started)[0:10]+' seconds to complete cropping')

#plt.scatter(zlon,zlat,c=var_crop[0]

if Chatty: zlon-zlon_wrf


# Do calculations to make wind total and rain total
#------------------------------------------

rainc,rainnc = np.asarray(var_crop_wrf[3]), np.asarray(var_crop_wrf[2])
rain_cumulative = [rainc[i] + rainnc[i] for i in range(len(rainc))]

rain = [[] for i in range(len(rain_cumulative))]

# remove the cumulative nature of rain variables
for i in range(len(rain_cumulative)):
  if i == 0: rain[0] = np.zeros(rain_cumulative[0].shape).tolist()
  else: rain[i] = (rain_cumulative[i]-rain_cumulative[i-1]).tolist()

# make total wind speed
v10,u10 = np.asarray(var_crop_wrf[-1]), np.asarray(var_crop_wrf[-2])
wind = [(v10[i]**2 + u10[i]**2)**.5 for i in range(len(rainc))]

var_crop_wrf = var_crop_wrf[0:7]

var_crop_wrf = var_crop_wrf + [rain] + [wind]
wrf_var = wrf_var + ['Rain', 'Wind10_TOT']
units_wrf = units_wrf + [units_wrf[2], units_wrf[-1]]


# Do calculations on cropped cmaq + wrf variables
#------------------------------------------

# get morning/midday/evening averages
hour = (np.array([6,7,8,9])+gmt_offset).tolist()
am_avg = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop,var)
am_avg_wrf = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop_wrf, wrf_var)

hour = (np.array([11,12,1+12,2+12])+gmt_offset).tolist()
mid_avg = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop,var)
mid_avg_wrf = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop_wrf, wrf_var)

hour = (np.array([5+12,6+12,7+12])+gmt_offset-24).tolist()
pm_avg = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop,var)
pm_avg_wrf = hourly_average(hour,np.arange(0,len(dates)).tolist(),var_crop_wrf, wrf_var)


#-----
# Create mask given Chicago shapefile

# chi_shapefile.contains(pt) get neighborhood values for each points

union=gpd.GeoSeries(unary_union(chi_shapefile.geometry)[2])

# routine to mask mask over chicago shapefile
mask=np.ones(zlon.shape,dtype=bool)
mask[:] = False

for i in range(len(zlon)):
    for j in range(len(zlon[0])):
       pt = Point(zlon[i][j],zlat[i][j])
       mask[i][j] =  pt.within(union[0])

# ------------ mask only for CHI
#[[total[i][j][~mask] = np.nan for i in range(len(total))] for j in range(len(total[0]))] 

for i in range(len(var_crop)):
   for j in range(len(var_crop[0])):
      var_crop[i][j][~mask] = np.nan 

var_crop_wrf = np.asarray(var_crop_wrf)
for i in range(len(var_crop_wrf)):
   for j in range(len(var_crop_wrf[0])):
      var_crop_wrf[i][j][~mask] = np.nan 


#make time series of days with EPA
#------------------------------------------
total= var_crop_wrf.tolist() + var_crop
#total= var_crop_wrf + var_crop

total = [var_crop_wrf[0]] + [var_crop_wrf[3]] +[var_crop_wrf[4]] + [var_crop_wrf[-1]] + var_crop

total = np.asarray(total)

titles = [wrf_var[0]] + [wrf_var[3]] +[wrf_var[4]] + [wrf_var[-1]] + var
#titles = wrf_var + var
hours = pd.date_range(dates[0]+" 00:00", dates[-2]+" 23:00",freq="60min")

# read in file
# avg_df = pd.read_csv('~/ChicagoStudy/T2_Q2_WIND_NO2_O3_CO_timeseries.csv')
# zlon,zlat?

# Across the model run, do average values follow a similar hourly pattern?
avg_df=pd.DataFrame(hours)
for w in range(len(total)):
   avg_daily=np.asarray([np.nanmean(np.asarray(total[w][i])) for i in range(len(total[0]))])
   avg_df[titles[w]]=avg_daily

avg_df['0']=pd.to_datetime(avg_df['0'])
avg_df = avg_df.T.drop('Unnamed: 0').T

#get hourly files
# get epa hourly ozone
epa_no2= pd.read_csv('/home/asm0384/ChicagoStudy/inputs/EPA_hourly_station_data/hourly_42602_2018.csv')
epa_no2 = epa_no2[epa_no2['State Name']=='Illinois']
epa_no2 = epa_no2[epa_no2['County Name'] == 'Cook']

epa_co = pd.read_csv('/home/asm0384/ChicagoStudy/inputs/EPA_hourly_station_data/hourly_42101_2018.csv')
epa_co.columns = epa_no2.columns
epa_co = epa_co[epa_co['State Name']=='Illinois']
epa_co = epa_co[epa_co['County Name'] == 'Cook']

#grep -hnr "Illinois" hourly_44201_2018.csv > hourly_44201_2018_ILLINOIS.csv
epa_o3 = pd.read_csv('/home/asm0384/ChicagoStudy/inputs/EPA_hourly_station_data/hourly_44201_2018_ILLINOIS.csv')
epa_o3.columns = epa_no2.columns
epa_o3 = epa_o3[epa_o3['State Name']=='Illinois']
epa_o3 = epa_o3[epa_o3['County Name'] == 'Cook']

#make daily averages
def epa_daily_average_from_hourly(epa_no2,hours):
   epa_no2['datetime'] = pd.to_datetime(epa_no2['Date GMT'] + ' ' + epa_no2['Time GMT'])
   #except: epa_no2['datetime'] = pd.to_datetime(epa_no2['Date Local'] + ' ' + epa_no2['Time GMT'])
   mask = (epa_no2['datetime'] >= hours[0]) & (epa_no2['datetime'] <= hours[-1])
   epa_no2= epa_no2.loc[mask]
   epa_lat,epa_lon= epa_no2['Latitude'].unique(),epa_no2['Longitude'].unique()
   epa_no2_avg = epa_no2.groupby('datetime')['Sample Measurement'].mean().reset_index()
   epa_no2_avg['std dev']=epa_no2.groupby('datetime')['Sample Measurement'].std().reset_index()['Sample Measurement']
   ours = pd.date_range(dates[0]+" 00:00", dates[-2]+" 23:00",freq="60min")
   epa_no2_avg= epa_no2_avg.set_index('datetime').reindex(hours,fill_value=np.nan).reset_index()
   return epa_no2_avg

epa_no2_avg = epa_daily_average_from_hourly(epa_no2,hours)
epa_o3_avg = epa_daily_average_from_hourly(epa_o3,hours)
epa_co_avg = epa_daily_average_from_hourly(epa_co,hours)

# Do the high values occur at the same time?
#max_df=pd.DataFrame(hours)
#for w in range(len(total)):
#   max_daily=np.asarray([np.nanpercentile(total[w][i],95) for i in range(len(total[0]))])
#   max_df[titles[w]]= max_daily

# Do the high lowest occur at the same time?
#min_df=pd.DataFrame(hours)
#for w in range(len(total)):
#   min_daily=np.asarray([np.nanpercentile(total[w][i],5) for i in range(len(total[0]))])
#   min_df[titles[w]]= min_daily


# Make heat maps of variables
sns.heatmap(avg_df.corr(),center = 0,annot = True)
plt.show()
sns.heatmap(max_df.corr(),center = 0, annot = True)
plt.show()
sns.heatmap(min_df.corr(),center = 0, annot=True)
plt.show()
sns.heatmap(max_df.corr()-avg_df.corr(),center = 0,annot=True)
plt.show()

# Make subplots of distributions of entire model run
fig,axs = plt.subplots(4,4, figsize=(10, 10))
axs= axs.ravel()

for c in range(16):
      ax = axs[c]
      ttttt = np.asarray(total[c])
      sns.distplot(ttttt.ravel()[~np.isnan(ttttt.ravel())],ax=ax,label = titles[c]).set_title(titles[c])

plt.tight_layout()
plt.show()

# Make time series of all days and highlight best/worst
wrst_df = avg_df
top10 = [[] for i in range(len(avg_df.columns)-1)]

for w in range(len(wrst_df.columns)):
   col = wrst_df.columns[w]
   if col == '0': pass
   else: top10[w-1] = avg_df.nlargest(n=10,columns=col)

bot10 = [[] for i in range(len(wrst_df.columns)-1)]

for w in range(len(wrst_df.columns)):
   col = wrst_df.columns[w]
   if col == '0': pass
   else: bot10[w-1] = wrst_df.nsmallest(n=10,columns=col)

fig, axs = plt.subplots(len(avg_df.columns)-1,1, figsize=(8, 10), facecolor='w', edgecolor='k')
axs=axs.ravel()

#avg_df['T2']=(avg_df['T2']-273.15)*9/5+32

for w in range(len(avg_df.columns)):
   if w == 0: pass
   else:
      epa = []
      ax=axs[w-1]
      if avg_df.columns[w] == 'O3': epa = epa_o3_avg['Sample Measurement']*1000
      elif avg_df.columns[w] == 'NO2': epa = epa_no2_avg['Sample Measurement']
      elif avg_df.columns[w] == 'CO': epa = epa_co_avg['Sample Measurement']*1000
      wrst=np.asarray(avg_df[avg_df.columns[w]])
      #ub,lb = [np.std(total[w][i])+ wrst[i] for i in range(len(total[0]))],[wrst[i] - np.std(total[w][i]) for i in range(len(var_crop_wrf[0]))]
      ax.plot(hours,wrst,color = 'purple')
      if len(epa)>0: ax.plot(hours,epa,color = 'crimson', linestyle = '--', alpha = 0.85)
      #ax.fill_between(hours, ub, lb, alpha=.5)
   #ax.plot(hours,[np.quantile(wrst,.75)]*len(wrst))
   #ax.plot(hours,[np.quantile(wrst,.25)]*len(wrst))
   #ax.scatter(top10[w][0], top10[w][titles[w]])
   #ax.scatter(bot10[w][0], bot10[w][titles[w]])
      ax.set_title(avg_df.columns[w])
      epa = []
      if w<len(avg_df.columns)-1: adjust_spines(ax,['left'])
      else: adjust_spines(ax,['left','bottom'])
      

sns.despine()
plt.tight_layout()
plt.show()

# now check high pollution episode
def filter_high_pollution(epa_no2_avg,h1,h2):
   #h1,h2 = hours[288], hours[360]
   try: mask = (epa_no2_avg['index'] >= h1) & (epa_no2_avg['index'] <= h2)
   except: mask = (epa_no2_avg['datetime'] >= h1) & (epa_no2_avg['datetime'] <= h2)
   epa_no2_avg= epa_no2_avg.loc[mask]
   return epa_no2_avg

h1,h2 = hours[288], hours[360]
epa_no2_avg_filter = filter_high_pollution(epa_no2_avg,h1,h2)
epa_o3_avg_filter = filter_high_pollution(epa_o3_avg,h1,h2)
epa_co_avg_filter = filter_high_pollution(epa_co_avg,h1,h2)

avg_df['0']=pd.to_datetime(avg_df['0'])
avg_df_filter = avg_df.loc[(avg_df['0'] >= h1) & (avg_df['0'] <= h2)]

pearsonr(avg_df_filter['NO2'], epa_no2_avg_filter['Sample Measurement'])
pearsonr(avg_df_filter['O3'], epa_o3_avg_filter['Sample Measurement'])
pearsonr(avg_df_filter['CO'], epa_co_avg_filter['Sample Measurement'])



from scipy.stats import pearsonr

# make 1:1 plots
fig,ax = plt.subplots(1,3,figsize = (12,8))

ax[0].scatter(epa_no2_avg_filter['Sample Measurement'], avg_df_filter['NO2'], color = 'purple',alpha =0.85)
ax[0].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['NO2'].min(), epa_no2_avg_filter['Sample Measurement'].min()]).min(), np.array([avg_df_filter['NO2'].max(), epa_no2_avg_filter['Sample Measurement'].max()]).max()
ax[0].set_xlim([int(min)-3,int(max)+3]); ax[0].set_ylim([int(min)-3,int(max)+3])
ax[0].set_aspect('equal')
ax[0].set_title('NO2, r = %.2f'%(pearsonr(avg_df_filter['NO2'], epa_no2_avg_filter['Sample Measurement'])[0],))

ax[1].scatter(epa_o3_avg_filter['Sample Measurement']*1000, avg_df_filter['O3'], color = 'crimson',alpha =0.85)
ax[1].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['O3'].min(), epa_o3_avg_filter['Sample Measurement'].min()*1000]).min(), np.array([avg_df_filter['O3'].max(), epa_o3_avg_filter['Sample Measurement'].max()*1000]).max()
ax[1].set_xlim([int(min)-3,int(max)+3]); ax[1].set_ylim([int(min)-3,int(max)+3])
ax[1].set_aspect('equal')
ax[1].set_title('O3, r = %.2f'%(pearsonr(avg_df_filter['O3'], epa_o3_avg_filter['Sample Measurement'])[0],))

ax[2].scatter(epa_co_avg_filter['Sample Measurement']*1000, avg_df_filter['CO'], color = 'orange',alpha =0.85)
ax[2].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['CO'].min(), epa_co_avg_filter['Sample Measurement'].min()*1000]).min(), np.array([avg_df_filter['CO'].max(), epa_co_avg_filter['Sample Measurement'].max()*1000]).max()
ax[2].set_xlim([int(min)-3,int(max)+3]); ax[2].set_ylim([int(min)-3,int(max)+3])
ax[2].set_aspect('equal')
ax[2].set_title('CO, r = %.2f'%(pearsonr(avg_df_filter['CO'], epa_co_avg_filter['Sample Measurement'])[0],))



# make 1:1 plots
fig,ax = plt.subplots(1,3,figsize = (12,8))


ax[0].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['NO2'].min(), epa_no2_avg_filter['Sample Measurement'].min()]).min(), np.array([avg_df_filter['NO2'].max(), epa_no2_avg_filter['Sample Measurement'].max()]).max()
ax[0].set_xlim([int(min)-3,int(max)+3]); ax[0].set_ylim([int(min)-3,int(max)+3])
ax[0].set_aspect('equal')
ax[0].set_title('NO2, r = %.2f'%(pearsonr(avg_df_filter['NO2'], epa_no2_avg_filter['Sample Measurement'])[0],))

ax[1].scatter(avg_df_filter['O3'], epa_o3_avg_filter['Sample Measurement']*1000,color = 'crimson',alpha =0.85)
ax[1].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['O3'].min(), epa_o3_avg_filter['Sample Measurement'].min()*1000]).min(), np.array([avg_df_filter['O3'].max(), epa_o3_avg_filter['Sample Measurement'].max()*1000]).max()
ax[1].set_xlim([int(min)-3,int(max)+3]); ax[1].set_ylim([int(min)-3,int(max)+3])
ax[1].set_aspect('equal')
ax[1].set_title('O3, r = %.2f'%(pearsonr(avg_df_filter['O3'], epa_o3_avg_filter['Sample Measurement'])[0],))

ax[2].scatter(avg_df_filter['CO'], epa_co_avg_filter['Sample Measurement']*1000, color = 'orange',alpha =0.85)
ax[2].plot([-1000,1000],[-1000,1000],color = 'black',alpha = 0.5)
min,max = np.array([avg_df_filter['CO'].min(), epa_co_avg_filter['Sample Measurement'].min()*1000]).min(), np.array([avg_df_filter['CO'].max(), epa_co_avg_filter['Sample Measurement'].max()*1000]).max()
ax[2].set_xlim([int(min)-3,int(max)+3]); ax[2].set_ylim([int(min)-3,int(max)+3])
ax[2].set_aspect('equal')
ax[2].set_title('CO, r = %.2f'%(pearsonr(avg_df_filter['CO'], epa_co_avg_filter['Sample Measurement'])[0],))







# start 9 figure cmaq plot to show diurnal
#-----------------------

crop_img_to_chicago = True #turn on or off if you want the boundary of image to follow shapefile

# get outside shape of shapefile to do plotting of the map
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry)[2])
outsideofunion=pd.DataFrame([list(union[0].exterior.xy)[0], list(union[0].exterior.xy)[1]])

# plots of maps
vars_thru_day = np.asarray([am_avg,mid_avg,pm_avg])
titles_top = ['6 - 9 AM', '11 AM - 2 PM', '5 - 7 PM']
dist_colors = sns.cubehelix_palette(3,start=.5, rot=-.75)
dist_colors = sns.color_palette("inferno_r",3) # reds and purples
yl=zlat.min()+.03;yu=zlat.max()-.03
xu=zlon.max()-.03;xl=zlon.min()+.03

crs_new = crs_new = ccrs.PlateCarree()


#''''''''''''''''''''''''''''''''
#fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(12, 10))
fig, axs = plt.subplots(len(var),4,figsize=(12, 10))

zz=0

vmin,vmax = [round(np.percentile(vars_thru_day[0][h].ravel(),5),4) for h in range(len(vars_thru_day[0]))],[round(np.percentile(vars_thru_day[0][h].ravel(),95),4) for h in range(len(vars_thru_day[0]))]

for w in range(len(vars_thru_day)):
   for zz in range(len(var)):
      #ax=plt.subplot(len(var),4,zz+w+1,projection = crs_new, frameon=False)
      ax = plt.subplot(axs[zz][w], subplot_kw={'projection': crs_new})
      #
      if crop_img_to_chicago == True: ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
      ax.add_geometries(Reader(lakemichigan).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
      ax.set_extent([xl,xu,yl,yu],crs= crs_new)
      cs=ax.pcolormesh(zlon,zlat,vars_thru_day[w][zz],transform=crs_new,cmap = 'inferno',vmin=vmin[zz],vmax=vmax[zz])
      chi_shapefile.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
      lm.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
      cbar=plt.colorbar(cs,boundaries=np.arange(vmin[zz],vmax[zz],round((vmax[zz]-vmin[zz])/5,5)),ax=ax,shrink=0.85)
      if zz == 0: ax.set_title(titles_top[w])
      #if zz*4+w+1 == 1 or zz*4+w+1 == 5 or zz*4+w+1 == 9:
      #   plt.ylabel(var[zz],ax=ax)
      #   #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
      #   cbar=plt.colorbar(cs,boundaries=np.arange(vmin[w],vmax[w]),ax=ax)
      #   cbar.set_ticks(np.arange(vmin[w], vmax[w], 10))

# distribution plots
for zz in range(len(var)):
   ax = axs[3][w]
   sns.distplot(am_avg[zz].ravel(), axlabel = cmaq_ncfile[0][var[zz]].units,label='am',color=dist_colors[0],ax= ax) # blue
   sns.distplot(mid_avg[zz].ravel(),label='mid', color=dist_colors[1],ax= ax) # orange
   sns.distplot(pm_avg[zz].ravel(),label = 'pm', color=dist_colors[2],ax= ax) #green
   if zz == 0: plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

plt.tight_layout()
sns.despine(bottom=True)
plt.show()


#-------------------
#hourly epa

from scipy.stats import pearsonr

ours = pd.date_range(dates[0]+" 00:00", dates[-1]+" 23:00",freq="60min")

epa_no2= pd.read_csv('no2_cook.csv')
#epa_no2 = epa_no2[epa_no2['State Name']=='Illinois']
#epa_no2 = epa_no2[epa_no2['County Name'] == 'Cook']
epa_no2['datetime'] = pd.to_datetime(epa_no2['Date GMT'] + ' ' + epa_no2['Time GMT'])

mask = (epa_no2['datetime'] >= hours[0]) & (epa_no2['datetime'] <= hours[-1])
epa_no2= epa_no2.loc[mask]
epa_lat,epa_lon= epa_no2['Latitude'].unique(),epa_no2['Longitude'].unique()

epa_no2_avg = epa_no2.groupby('datetime')['Sample Measurement'].mean().reset_index()
epa_no2_avg['std dev']=epa_no2.groupby('datetime')['Sample Measurement'].std().reset_index()['Sample Measurement']

var= ['NO2']

cmaqncfile_vars = [np.array([cmaq_ncfile[j][var[i]] for i in range(len(var))]).mean(axis=0) for j in range(len(cmaq_ncfile))]

cmaqncfile_vars6 = [[cmaq_ncfile[j][var[i]] for i in range(len(var))] for j in range(len(cmaq_ncfile))]

cmaqncfile_vars6 = [cmaq_ncfile[j][var[0]] for j in range(len(cmaq_ncfile))]
cmaqncfile_vars67 = [np.array(cmaqncfile_vars6[i]).mean(axis=0)[0] for i in range(len(cmaq_ncfile))]

#list,array
ind_stn = find_index(epa_lon.tolist(), epa_lat.tolist(), lon, lat)
stnpixel_from_cmaq=[[var_crop[1][i][ind_stn[0][f]][ind_stn[1][f]] for i in range(len(var_crop[1]))] for f in range(len(ind_stn[0]))]

stnpixel_from_cmaq = [[cmaqncfile_vars67[i][ind_stn[0][j]][ind_stn[1][j]] for j in range(len(ind_stn[0]))] for i in range(len(cmaqncfile_vars67))]


#get hourly measurements with all timesteps
stn_epa = [epa_no2[epa_no2['Latitude']== epa_lat[i]].reset_index() for i in range(len(epa_lat))]
stn_epa = [stn_epa[i].set_index('datetime').reindex(hours,fill_value=np.nan).reset_index() for i in range(len(epa_lat))]
stn_epa = [stn_epa[i]['Sample Measurement'].tolist() for i in range(len(epa_lat))]

d = epa_no2.set_index('datetime')
a = d.groupby('Latitude').resample('D').mean()
stn_epa_mix = [pd.DataFrame(a['Sample Measurement'][epa_lat[i]]).reindex(ours,fill_value=np.nan).reset_index() for i in range(len(epa_lat))]

plt.scatter(stn_epa_mix[0]['Sample Measurement'], pd.DataFrame(stnpixel_from_cmaq)[0])
plt.scatter(stn_epa_mix[1]['Sample Measurement'], pd.DataFrame(stnpixel_from_cmaq)[1])
plt.scatter(stn_epa_mix[2]['Sample Measurement'], pd.DataFrame(stnpixel_from_cmaq)[2])


# get 1pm measurements
epa_no2_1pm = epa_no2[epa_no2['Time GMT'] == '06:00']
epa_no2_1pm = epa_no2_1pm[epa_no2_1pm['Site Num'] == 4002]

# pull PM over time
fig = plt.figure(figsize = (7,5))
plt.plot(hours, stn_epa[2],label = 'Station', color = 'black')
plt.plot(hours, stnpixel_from_cmaq[2], dashes=[6, 2], label = 'Model',alpha = 0.5, color = 'black')
plt.scatter(epa_no2_1pm['datetime'],epa_no2_1pm['Sample Measurement'], color = 'red',marker = '*',s=300,label = '1 PM Overpass')
plt.xticks(rotation = 60)
plt.legend()
plt.tight_layout()
sns.despine()
plt.show()

#plot hourly concentrations 
#plt.plot(hours,stn_epa[0],color='black')
#plt.plot(hours,stn_epa[1],color='black')
#plt.plot(hours,stn_epa[2],color='black')
plt.plot(hours, stnpixel_from_cmaq[1])
plt.plot(hours, stnpixel_from_cmaq[2])

max,min = int(np.nanmax([np.nanmax(stnpixel_from_cmaq),np.nanmax(stn_epa)])),int(np.nanmin([np.nanmin(stnpixel_from_cmaq),np.nanmin(stn_epa)]))

min=0

fig, axs = plt.subplots(2,3,figsize=(12, 10))

for i in range(len(stn_epa)):
   ax=axs.ravel()[i]
   tmp_df = pd.DataFrame([stn_epa[i],stnpixel_from_cmaq[i]]).T.dropna()
   sns.distplot(tmp_df[0],ax=ax, label = 'EPA', axlabel=None),# kde_kws={"lw": 3, "label": "EPA"})
   sns.distplot(tmp_df[1],ax=ax,label = 'CMAQ', axlabel=None),# kde_kws={"lw": 3, "label": "CMAQ"})
   ax.legend()
   ax.set_xlabel('NO2 Conc')
   ax.set_xlim(min,max)
   ax.set_title(epa_no2['Site Num'].unique()[i])
   ax= axs.ravel()[i+3]
   sns.lineplot([min,max],[min,max],color = 'grey',alpha = 0.5, ax=ax)
   sns.scatterplot(tmp_df[0],tmp_df[1],ax=ax,color='red')
   #sns.regplot(tmp_df[0],tmp_df[1],ax=ax,color='black')
   ax.set_title('R = ' +str(pearsonr(tmp_df[0],tmp_df[1])[0])[0:5],loc = 'Right')
   ax.set_xlabel('EPA ppb')
   ax.set_ylabel('CMAQ ppb')
   ax.set_xlim(min,max)
   ax.set_ylim(min,max)

sns.despine()
plt.show()


del tmp_df
pearsonr(stnpixel_from_cmaq[0], stn_epa[0])



# map of epa station points
chi_shapefile.plot(facecolor='None',edgecolor='black')
plt.scatter(epa_lon,epa_lat,s = 350,color = 'red')
#plt.show()

# relate spatial pixels to eachother
#------------------------------------------

from scipy.stats import pearsonr

vca = np.asarray(var_crop)
tmpa = np.asarray(var_crop_wrf)
vca_am,tmpa_am = np.asarray(am_avg), np.asarray(am_avg_wrf)
vca_mid,tmpa_mid = np.asarray(mid_avg), np.asarray(mid_avg_wrf)
vca_pm,tmpa_pm = np.asarray(pm_avg), np.asarray(pm_avg_wrf)

total2 = total.copy()

for i in range(len(total)):
   for j in range(len(total[0])):
      total[i][j][~mask] = 8000 

total = [total[y][0:len(hours)] for y in range(len(total))]
total = np.asarray(total)

#array of wrf variables to conc of pollutants
temp_o3=[[pearsonr(total[0][:,i,j],total[7][:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]

every_map_corr = [[[[pearsonr(total[u][:,i,j],total[uk][:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))] for u in range(len(total))] for uk in range(len(total))]

every_map_corr_title = [[titles[u]+ 'vs' + titles[uk] for u in range(len(total))] for uk in range(len(total))]

# Make plots for every correlation neighbs 
#------------------
crs_new = ccrs. AlbersEqualArea(central_longitude=(chi_shapefile.bounds.mean().minx+chi_shapefile.bounds.mean().maxx)/2)
# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])
yl=zlat.min()+.03;yu=zlat.max()-.03
xu=zlon.max()-.03;xl=zlon.min()+.03

crop_img_to_chicago= True

for i in range(len(total)):
   for j in range(len(total)):
       vmin,vmax = np.nanpercentile(every_map_corr[j][i],5),np.nanpercentile(every_map_corr[j][i],95)
       fig, ax = plt.subplots(subplot_kw={'projection': crs_new},figsize=(12, 10))
       if crop_img_to_chicago == True: ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
       ax.add_geometries(Reader(lakemichigan).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
       ax.set_extent([xl,xu,yl,yu],crs= crs_new)
       cs=ax.pcolormesh(zlon,zlat, every_map_corr[j][i],transform=crs_new,cmap = 'inferno',vmin=vmin,vmax=vmax)
       chi_shapefile.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
       lm.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
       cbar=plt.colorbar(cs,boundaries=np.arange(vmin,vmax,(vmax-vmin)/10),ax=ax,shrink=0.85)
       ax.set_title(every_map_corr_title[j][i])
       plt.savefig(every_map_corr_title[j][i]+'.pdf')
 

temp_no2=[[pearsonr(vca[1,:,i,j],tmpa[0,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]
temp_pm=[[pearsonr(vca[2,:,i,j],tmpa[0,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]

psfc_o3=[[pearsonr(vca[0,:,i,j],tmpa[1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]
psfc_no2=[[pearsonr(vca[1,:,i,j],tmpa[1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]
psfc_pm=[[pearsonr(vca[2,:,i,j],tmpa[1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]

rh_o3=[[pearsonr(vca[0,:,i,j],tmpa[-1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]
rh_no2=[[pearsonr(vca[1,:,i,j],tmpa[-1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]
rh_pm=[[pearsonr(vca[2,:,i,j],tmpa[-1,:,i,j])[0] for j in range(len(var_crop[0][0][0]))] for i in range(len(var_crop[0][0]))]

for zz in range(len(var_crop)):
   var_crop_0 = pd.DataFrame([var_crop[0][i].ravel() for i in range(len(var_crop[0]))]).T
#sns.heatmap(var_crop_0.corr())
   averages[titles[zz]] = var_crop_0.mean(axis=1)

extreme = [np.nanpercentile(var_crop_0[i],95) for i in range(len(var_crop_0.T))] 
averages = [np.nanmean(var_crop_0[i]) for i in range(len(var_crop_0.T))]
var_crop_1 = pd.DataFrame([var_crop[1][i].ravel() for i in range(len(var_crop[1]))]).T
extreme1 = [np.nanpercentile(var_crop_1[i],95) for i in range(len(var_crop_1.T))] 
averages1 = [np.nanmean(var_crop_1[i]) for i in range(len(var_crop_1.T))]


#all corr plots
all_wrf=[temp_o3, temp_no2, temp_pm, psfc_o3, psfc_no2, psfc_pm, rh_o3, rh_no2, rh_pm]


fig, axs = plt.subplots(3,3,subplot_kw={'projection': crs_new},figsize=(12, 10))

vmin,vmax=0,1

for w in range(len(all_wrf)):
      if w<3: ax = axs[0][w]
for z in range(1):
      if w<6 and w>=3: ax = axs[1][3-w]
      if w>=6: ax = axs[2][6-w]
      if crop_img_to_chicago == True: ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
      ax.add_geometries(Reader(lakemichigan).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
      ax.set_extent([xl,xu,yl,yu],crs= crs_new)
      cs=plt.pcolormesh(zlon,zlat,all_wrf[w],transform=crs_new,cmap = 'inferno',vmin=vmin,vmax=vmax)
      chi_shapefile.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
      lm.plot(ax=ax, transform= crs_new,facecolor='None',edgecolor='black',alpha=0.3)
      #cbar=plt.colorbar(cs,boundaries=np.arange(int(vmin),int(vmax)),ax=ax,shrink=0.85)
      if zz == 0: plt.title(titles_top[w])
      #if w+1 == 1 or w+1 == 5 or w+1 == 9:
      #   plt.ylabel(var[zz],ax=ax)
      #   #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
      #   cbar=plt.colorbar(cs,boundaries=np.arange(vmin[w],vmax[w]),ax=ax)
      #   cbar.set_ticks(np.arange(vmin[w], vmax[w], 10))

# distribution plots
for zz in range(len(var)):
   plt.subplot(3,4,zz*4+4)
   sns.distplot(am_avg[zz].ravel(), axlabel = cmaq_ncfile[0][var[zz]].units,label='am',color=dist_colors[0]) # blue
   sns.distplot(mid_avg[zz].ravel(),label='mid', color=dist_colors[1]) # orange
   sns.distplot(pm_avg[zz].ravel(),label = 'pm', color=dist_colors[2]) #green
   if zz == 0: plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

plt.tight_layout(h_pad=1.9)
sns.despine(bottom=True)
plt.show()

# get averages of temps on the hour
vca_avg = [[vca[j][i].mean() for i in range(len(vca[0]))] for j in range(len(vca))]
vca_wrf_avg = [[tmpa[j][i].mean() for i in range(len(tmpa[0]))] for j in range(len(tmpa))]

vca_ravel = [[vca[j][i].ravel() for i in range(len(vca[0]))] for j in range(len(vca))]
vca_wrf_ravel= [[tmpa[j][i].ravel() for i in range(len(tmpa[0]))] for j in range(len(tmpa))]


# heat map of related pollutants
[var_crop[i] for i in range(len(var))]+ [var[i]+'_AM' for i in range(len(var))]+[var[i]+'_MID' for i in range(len(var))]+[var[i]+'_PM' for i in range(len(var))]+[wrf_var[i] for i in range(len(var))]+[wrf_var[i]+'_AM' for i in range(len(wrf_var))]+[wrf_var[i]+'_MID' for i in range(len(wrf_var))]+[wrf_var[i]+'_PM' for i in range(len(wrf_var))]

cols = [var[i] for i in range(len(var))]+ [var[i]+'_AM' for i in range(len(var))]+[var[i]+'_MID' for i in range(len(var))]+[var[i]+'_PM' for i in range(len(var))]+[wrf_var[i] for i in range(len(var))]+[wrf_var[i]+'_AM' for i in range(len(wrf_var))]+[wrf_var[i]+'_MID' for i in range(len(wrf_var))]+[wrf_var[i]+'_PM' for i in range(len(wrf_var))]




#   forgotten plot
#------------------------------------------
"
i,j=0,0

var_crop_wrf_hourly_average = [[np.average(var_crop_wrf[j][i]) for i in range(len(var_crop_wrf[0]))] for j in range(len(var_crop_wrf))]

i,j=0,0
var_crop_hourly_average = [[np.average(var_crop[j][i]) for i in range(len(var_crop[0]))] for j in range(len(var_crop))]

hours = pd.date_range(dates[0]+" 00:00", dates[-1]+" 23:00",freq="60min")


fig, axes = plt.subplots(len(var_crop_wrf_hourly_average)+ len(var_crop_hourly_average), 1, figsize=(10,8), sharex=True)
for i in range(len(var_crop_wrf_hourly_average)+ len(var_crop_wrf_hourly_average)-2):
   ax=axes[i]
   if i<len(var_crop_wrf_hourly_average): sns.lineplot(x= hours,y= var_crop_wrf_hourly_average[i],ax=ax)
   else: sns.lineplot(x= hours,y= var_crop_hourly_average[i-len(var_crop_wrf_hourly_average)],ax=ax)
   plt.ylabel(ylabel=t[i])

plt.show()

sns.despine(bottom=True)
"
fig, axes = plt.subplots(len(var_crop_wrf_hourly_average)+ len(var_crop_hourly_average), 1, figsize=(10,8), sharex=True)
for i in range(len(var_crop_wrf_hourly_average)+ len(var_crop_wrf_hourly_average)-2):
   ax=axes[i]
   if i<len(var_crop_wrf_hourly_average): sns.lineplot(x= hours,y= var_crop_wrf_hourly_average[i],ax=ax)
   else: sns.lineplot(x= hours,y= var_crop_hourly_average[i-len(var_crop_wrf_hourly_average)],ax=ax)
   plt.ylabel(ylabel=t[i])

plt.show()

sns.lineplot(x= np.asarray(var_crop_wrf[0]).ravel(),y= np.asarray (var_crop[0]).ravel())

sns.heatmap(x= var_crop_wrf, y = var_crop)

#------------------------------------------#------------------------------------------
#------------------------------------------#------------------------------------------
# Change in emissions scenario
#------------------------------------------
# files
dir='/projects/b1045/jschnell/ForStacy/'
ll='latlon_ChicagoLADCO_d03.nc'
emis='emis_20180801_noSchoolnoBusnoRefuse_minus_base.nc'
#emis='emis_20180801_noSchool_minus_base.nc'
ll=Dataset(dir+ll,'r')
lat,lon=ll['lat'][:],ll['lon'][:]

path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'

# Start pulling and cropping data
chi  = gpd.GeoDataFrame.from_file(path)


#pull in files and variables
ncfile= Dataset(dir+emis,'r')
df_lat,df_lon=pd.DataFrame(lat),pd.DataFrame(lon)
no2= pd.DataFrame(Dataset(dir+emis,'r')['NO2'][13][0][:])*10e2
no2 = pd.DataFrame(Dataset(dir+emis,'r')['NO'][13][0][:])*10e2
df=pd.DataFrame(no2[:])

#find all rows and columns where the change is 0 and drop them
no2_drop=df.loc[~(df==0).all(axis=1)]

# given where no2 values are 0, filter out the lat lons
# ie. drop the outside parts that are 0 change in the array
data= np.array(df.loc[~(df==0).all(axis=1)])
lat= np.array(df_lat.loc[~(df==0).all(axis=1)])
lon= np.array(df_lon.loc[~(df==0).all(axis=1)])


# files
emis1='emis_20180801_noSchoolnoBusnoRefuse_minus_base.nc'

#pull in files and variables
ncfile1= Dataset(dir+emis1,'r')

no21= pd.DataFrame(Dataset(dir+emis1,'r')['NO'][13][0][:])*10e2
df1=pd.DataFrame(no21[:])

# drop outside parts that are 0 in the array
data1= np.array(df1.loc[~(df1==0).all(axis=1)])
lat1= np.array(df_lat.loc[~(df1==0).all(axis=1)])
lon1= np.array(df_lon.loc[~(df1==0).all(axis=1)])
#data=data-data1


crs_new = ccrs. AlbersEqualArea(central_longitude=(chi_shapefile.bounds.mean().minx+chi_shapefile.bounds.mean().maxx)/2)

data = np.asarray(var_crop[1]).mean(axis=0)
titl = var[1] + ' Mean over August 2018'
lbl = 'ppbV'
#data=(data-273.15) * 9/5 + 32 


data = np.asarray(temp_no2)*-1
#data = np.asarray(var_crop_wrf[0]).max(axis=0)
#data=(data-273.15) * 9/5 + 32 

# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])

# make fig object
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 6))

#set up data for plotting via levels
vmax= pd.DataFrame(data).max().max()
vmin= int(pd.DataFrame(data).min().min())+.5
#vmax=10
#vmin=0
levels = np.linspace(vmin,(vmax), 11)


# get rid of values outside the levels we are contouring to
#data[pd.DataFrame(data)<vmin]=vmin


# set boundary as outer extent by making a matplotlib path object and adding that geometry
# i think setting the boundary before you plot the data actually crops the data to the shape, so set ax first
axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')

#plot the gridded data by using contourf
#cs=plt.contourf(zlon,zlat,data,cmap= "inferno_r", transform=crs_new, levels=levels)
cs = plt.pcolormesh(zlon,zlat,data,transform=crs_new,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)

# add landmarks with scatterplot
midway= 41.7868, -87.7522
ohare = 41.9742, -87.9073
loop = 41.8786, -87.6251
plt.scatter(pd.DataFrame([midway,ohare,loop])[1],pd.DataFrame([midway,ohare,loop])[0],marker = '*',color='white')

# set axes extents from shapefile
x=[chi_shapefile.bounds.minx.min(), chi_shapefile.bounds.maxx.max()] 
y=[chi_shapefile.bounds.miny.min(), chi_shapefile.bounds.maxy.max()]
axs.set_extent([x[0]-.03,x[1]+.03,y[0]-.03,y[1]+.03],crs= crs_new)
axs.set_title(titl)


#add colorbar and label
cbar=plt.colorbar(cs,boundaries=levels, shrink=0.7)
cbar.ax.set_ylabel(lbl)
cbar.set_ticks(levels)


plt.show()


#-----
# Create mask given Chicago shapefile

# chi_shapefile.contains(pt) get neighborhood values for each points

union=gpd.GeoSeries(unary_union(chi_shapefile.geometry)[2])

# routine to mask mask over chicago shapefile
mask=np.ones(zlon.shape,dtype=bool)
mask[:] = False

for i in range(len(zlon)):
    for j in range(len(zlon[0])):
       pt = Point(zlon[i][j],zlat[i][j])
       mask[i][j] =  pt.within(union[0])




