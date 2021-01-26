#!/bin/bash python3

#---------------------------------------------------------#
# Stacy Montgomery, Aug 2019
# Purpose: find aqs stations within model domain, 
#          pull & format aqs data for comparison
# 

# Link to air tech website with year you're interested in -- NOT WORKING
#linktoaqs='http://files.airnowtech.org/?prefix=airnow/2018/'
# USE: 
#---------------------------------------------------------#

# LIBRARIES
#---------------------------------------------------------#
from datetime import timedelta, date,datetime; import pandas as pd
import numpy as np
from netCDF4 import Dataset
from wrf import latlon_coords, getvar
import glob, os
import matplotlib.pyplot as plt

#import requests
#from bs4 import BeautifulSoup


# USER INPUT
#---------------------------------------------------------#
# Find stations within bounding box 
#llon,llat,ulon,ulat=-98.854465,39.517152,-74.289036,49.678626 #use bounds from griddesc

# Date range to pull from AQS --- if commented out, defined by cmaq files avail
#start_dt = date(2018, 8, 1); end_dt = date(2018, 9, 1)


# Directories for cmaq + EPA 
dir_epa='/home/asm0384/CMAQcheck/'
grid='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d02/lat_lon_chicago_d02.nc'
#grid='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d03/latlon_ChicagoLADCO_d03.nc'

dir_cmaq='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/postprocess/'
dir_cmaq='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/postprocess/'

# to get grid, pull WRF coords
#runname='wrf_pure_PXLSM'
#dirToWRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/'+runname+'/' # to get grid

# CMAQ RUN things
domain='d02'
time='hourly'
year='2019'
month='1'
#epa_code=['42401','42602','44201','42101']; var=['SO2','NO2','O3','CO'] #numerical identifiers and corresponding vars
#epa_code=['44201'];var=['O3']
epa_code=['88101']
var = ['PM25_TOT']
epa_files =[dir_epa+'%s_%s_%s.csv'%(time,epa_code[i],year,) for i in range(len(epa_code))]

#names of lat lons in the cmaq grid
la,lo='lat','lon' # for 1.3km
la,lo='LAT','LON' # for 4km

# USER DEF FUNC
#---------------------------------------------------------#


#------ DATERANGE
# 
#
# * dates must be in yyyymmdd format
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

#------ VARfromIND
# 
#
def getVARfromIND(ncfile,indxy, filenames,varname):
    t2d01=[ncfile[z][varname][i] for z in range(len(filenames)) for i in range(24)]
    t2d01_xx=  [[t2d01[t][indxy[l]] for t in range(24*len(filenames_d01))] for l in range(len(indxy))]
    return t2d01_xx

#------ FIND INDEX
# 
#
# adapted from : http://kbkb-wx-python.blogspot.com/2016/08/find-nearest-latitude-and-longitude.html
def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
# stn -- points 
# wrf -- list
#for iz in range(1):
   xx=[];yy=[]
   for i in range(len(stn_lat)):
   #for i in range(1):
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

#------ PULL CMAQ
# 

def pull_cmaq(file, start_dt, end_dt,llat,ulat,llon,ulon,cmaq_lon,cmaq_lat,cmaq,VAR):
   #read in file
   f=pd.read_csv(file)
   # Crop given bounding box
   df=f[(f['Latitude'] >=  llat) & (f['Latitude'] <=  ulat)]
   df=df[(df['Longitude'] >=  llon) & (df['Longitude'] <=  ulon)]
   df['Datetime GMT']=pd.to_datetime(df['Date GMT']+ ' ' + df['Time GMT'])
   df= df[(df['Datetime GMT'] >= pd.to_datetime(start_dt) ) & (df['Datetime GMT'] <= pd.to_datetime(end_dt))]
   lon,lat=df['Longitude'].unique(),df['Latitude'].unique()
   df.reset_index(inplace=True)
   return lon,lat,df
#somehow make the 0s match up


#------ RESAMPLE DF
# Take in real data, fill in missing values with missing values but keep that date open
#
def resample_df(df,lat,lon,start_dt,end_dt):
   dff=pd.DataFrame()
   # get list of target dates
   t_index = pd.DatetimeIndex(start=start_dt, end=end_dt, freq='1h')
   #set index as dates
   df.set_index('Datetime GMT',inplace=True)
   # go through each locations and fill in missing dates
   for i in range(len(lat)):
      check=df[(df['Latitude']==lat[i]) & (df['Longitude']==lon[i])]
      #aka: if there are multiple sensors of same thing, just average
      if len(check['POC'].unique())>1 or len(check) > len(t_index):
         sample = check.resample('H').mean().reindex(t_index).fillna(float('nan'))['Sample Measurement']
         df2=check[check['POC']==1].resample('H').asfreq().reindex(t_index).fillna(float('nan'))
         df2['Sample Measurement']=sample
         #print('%s in %s,%s is irregular'%(check['Site Num'][0] ,check['County Name'][0],check['State Name'][0],))
      else: #just fill out values
         df2 = check.resample('H').asfreq().reindex(t_index).fillna(float('nan'))
      #averaged or not, add to final df
      dff=dff.append(df2)
   #return index with index rather than dates
   dff.reset_index(inplace=True)
   return dff

--------------------------------------------------#

# $1 Get CMAQ file names
cmaq_files=[]
os.chdir(dir_cmaq)
for file in glob.glob("COMBINE_ACONC_*"):
    cmaq_files.append(file)

cmaq_files.sort()
cmaq_files.remove(cmaq_files[-1])

dates=[cmaq_files[z].split("COMBINE_ACONC_")[1].split(".nc")[0] for z in range(len(cmaq_files))]
start_dt=date(int(dates[0][0:4]),int(dates[0][4:6]),int(dates[0][6:8]))
end_dt=datetime(int(dates[-1][0:4]),int(dates[-1][4:6]),int(dates[-1][6:8]),23,0)

# Get first date range, pull monitoring station range ... unecessary
#dtrange=[]
#for dt in daterange(start_dt, end_dt):
#    dtrange.append(dt.strftime("%Y%m%d"))

# Get cmaq grid
#grid='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d02/GRIDCRO2D_Chicago_LADCO_2018-08-20.nc'
#cmaq_lat,cmaq_lon=Dataset(grid)['LAT'][0][0],Dataset(grid)['LON'][0][0]

cmaq_lat,cmaq_lon = np.asarray(Dataset(grid)[la]),np.asarray(Dataset(grid)[lo])

if len(cmaq_lat.shape) == 4:
   cmaq_lat,cmaq_lon = cmaq_lat[0][0],cmaq_lon[0][0]

print('CMAQLATSHAPE')
print(cmaq_lat.shape)
#cmaq_lat,cmaq_lon = np.asarray(Dataset(grid)['LAT'])[0][0],np.asarray(Dataset(grid)['LON'])[0][0]
llat,ulat,llon,ulon=cmaq_lat.min(), cmaq_lat.max(), cmaq_lon.min(), cmaq_lon.max()

cmaq=[Dataset(dir_cmaq+cmaq_files[i]) for i in range(len(cmaq_files))]
t_index = pd.DatetimeIndex(start=start_dt, end=end_dt, freq='1h')

# Loop through each variable and check
for loop in range(len(epa_files)):
   lon,lat,df=pull_cmaq(epa_files[loop], start_dt, end_dt, llat,ulat,llon,ulon,cmaq_lon,cmaq_lat,cmaq,var[loop])
   xx,yy= find_index(lon, lat, cmaq_lon, cmaq_lat)
   dff= resample_df(df,lat,lon,start_dt,end_dt)
   dff['CMAQ']=float('nan')
   for numday in range(len(cmaq)):
      s=pd.DataFrame([[cmaq[numday][var[loop]][time][0][xx[idx]][yy[idx]] for time in range(24)] for idx in range(len(xx))]).T
      #
      for station in range(len(xx)):
            dff['CMAQ'][(24*(numday)+station*len(t_index)):(24*(numday)+ station*len(t_index)+24)]=s[station]
   # Output the var
   dff.to_csv(dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[loop],domain,year,month));
   print('Done with %s'%(var[loop]));

#end





