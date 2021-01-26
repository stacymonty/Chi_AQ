!/bin/bash python3

# Stacy Montgomery, April 2019
#Single day WRF output

# -----------------
# Step 1
# -----------------

# Data for comparison
# LCD data from noaa: https://www.ncei.noaa.gov/data/local-climatological-data/access/2018/
# LCD station names -- metadata of stations -- must make CSV: https://www.ncdc.noaa.gov/homr/file/lcd-stations.txt
# Currently the UTC offset calculator is for negative offsets, simple loop fix to do positive offsets
# ---------------------------------------------------------------------------------------------------------
# ~~~~~~ START USER INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
monthNum=[i for i in range(12)]
daysOfMonths=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
year='2019'
month='01'

# Location of US Data
NOAAdataLink="https://www.ncei.noaa.gov/data/local-climatological-data/access/"+year+"/"
listOfStationsFile="~/lcd-stations.csv" #metadata of stations

# Name of run
runname='output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852'
#BASE_PXLSM_v0
# Location of WRF output

dirToWRF_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/'
dirToWRF_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/'
listOfStationsFile = "~/lcd-stations.csv"
dirout='/home/asm0384/WRFcheck/'+runname+'/'
grid='latlon.nc'
#
Chatty= True       # false if you want to remove print statements
written= True      #if the station files have been filtered for d01 and missing links (the function findStations() has previously been run)

#File out names
comp_dataset_name = 'wrfcheck_withstations_'+runname+'_'+month+year+'.csv'                     # name and directory to write out to
comp_dataset_extra = 'completeddata_mini_extras2.csv'
station_out_name = 'station_out_removedmissing.csv' #name of intermediate file
comp_dataset_name2= 'wrfcheck_withstations_complete_rain.csv'

four=['SLP', 'RH', 'WindDir', 'Wind']
comp_dataset_name_list_of4=['wrfcheck_withstations_'+runname+'_'+four[i]+'.csv' for i in range(len(four))]

if Chatty: print('Starting ....')

# --------------------------------------------------------------------------------------------------------
# ~~~~~~ IMPORT PACKAGES ~~~~~~~~~~~~
#Station
import glob, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt, cartopy.crs as crs, cartopy.feature as cpf
from netCDF4 import Dataset
from matplotlib.cm import get_cmap
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import time
from timezonefinder import TimezoneFinder
from pytz import timezone
import pytz
from datetime import datetime,date, timedelta
import dateutil.parser as dparser

tf = TimezoneFinder(in_memory=True)

# ~~~~~~ CUSTOM FUNCTIONS ~~~~~~~~~~~~
# adapted from : http://kbkb-wx-python.blogspot.com/2016/08/find-nearest-latitude-and-longitude.html
def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
# stn -- points 
# wrf -- list
   xx=[];yy=[]
   for i in range(len(stn_lat)):
      abslat = np.abs(wrf_lat-stn_lat[i])
      abslon= np.abs(wrf_lon-stn_lon[i])
      c = np.maximum(abslon,abslat)
      x=np.where(c==c.min())
      if x[0][0]==0 and x[1][0]==0 and len(x)>2: x,y=x[2],x[3]
      else: x,y = x[0],x[1]
      #add indices of nearest wrf point station
      xx.append(x)
      yy.append(y)
   #return indices list
   return xx, yy
                                                                
                                                                

# modified from https://stackoverflow.com/questions/16685384/finding-the-indices-of-matching-elements-in-list-in-python
def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

#  ----------------------------------------------------
utc = pytz.utc
def offset(lat,lon):
    #returns a location's time zone offset from UTC in minutes.
    today = datetime.now()
    tz_target = timezone(tf.certain_timezone_at(lat=lat, lng=lon))
    # ATTENTION: tz_target could be None! handle error case
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return (today_utc - today_target).total_seconds() / 3600

#  ----------------------------------------------------
# pull in real data, apply UTC, and average and remove hourly values
def getRealData(LCD):
   big_start=time.time()
   date_noTime=[]; time_noDate=[]
   date_noTime= [LCD['DATE'][z].split('T')[0] for z in range(len(LCD['DATE']))]
   time_noDate=[LCD['DATE'][z].split('T')[1] for z in range(len(LCD['DATE']))]
   UTC_offset=-offset(lon=LCD['LONGITUDE'][0], lat=LCD['LATITUDE'][0])
   #get day before and after for UTC offset sake
   date_onedaybefore=(dparser.parse(dates[0])-timedelta(days=1)).isoformat().split('T')[0]
   date_onedayafter=(dparser.parse(dates[-1])+timedelta(days=1)).isoformat().split('T')[0]
   start_ind_dataset = find(date_noTime, date_onedaybefore)[0]
   end_ind_dataset= find(date_noTime, date_onedayafter)[-1]
   if Chatty: print('-> Adding UTC offset to timestamp and averaging repeated values (%.2f)'%(time.time()-big_start,))
   # UTC offset calculator
   # Get the time and round up or round down, also add the UTC offset such that correct time is in UTC
   correctedTime=[];correctedSLP=[]; correctedWindDir=[]; correctedRH=[]; correctedWind =[]; correctedRain=[]; correctedTemp =[];correctedDate=[]
   for i in range(len(LCD[start_ind_dataset: end_ind_dataset])):
      datetimeLCD=dparser.parse(LCD['DATE'][start_ind_dataset+i])
      datetimeLCD_UTC = datetimeLCD + timedelta(hours=UTC_offset)
      try:
         rainz = float(LCD['HourlyPrecipitation'][start_ind_dataset+i])
      except ValueError:
         rainz =float('nan')
      try:
         tempz= float(LCD['HourlyDryBulbTemperature'][start_ind_dataset+i])
      except ValueError:
         tempz=float('nan')
      try:
         slpz= float(LCD['HourlySeaLevelPressure'][start_ind_dataset+i])
      except ValueError:
         slpz=float('nan')
      try:
         windz= float(LCD['HourlyWindSpeed'][start_ind_dataset+i])
      except ValueError:
         windz=float('nan')
      try:
         RHz= float(LCD['HourlyRelativeHumidity'][start_ind_dataset+i])
      except ValueError:
         RHz=float('nan')
      try:
         WindDirz= float(LCD['HourlyWindDirection'][start_ind_dataset+i])
      except ValueError:
         WindDirz=float('nan')
      if datetimeLCD_UTC.minute >= 30:
            correctedTime.append((datetimeLCD_UTC+timedelta(minutes=60-datetimeLCD_UTC.minute)).isoformat().split('T')[1])
            correctedDate.append((datetimeLCD_UTC+timedelta(minutes=60-datetimeLCD_UTC.minute)).isoformat().split('T')[0])
            correctedRain.append(rainz)
            correctedTemp.append(tempz)
            correctedSLP.append(slpz)
            correctedWind.append(windz)
            correctedRH.append(RHz)
            correctedWindDir.append(WindDirz)
      elif datetimeLCD_UTC.minute < 30:
            correctedTime.append((datetimeLCD_UTC+timedelta(minutes=-datetimeLCD_UTC.minute)).isoformat().split('T')[1])
            correctedDate.append((datetimeLCD_UTC+timedelta(minutes=-datetimeLCD_UTC.minute)).isoformat().split('T')[0])
            correctedRain.append(rainz)
            correctedTemp.append(tempz)
            correctedSLP.append(slpz)
            correctedWind.append(windz)
            correctedRH.append(RHz)
            correctedWindDir.append(WindDirz)
   #Now filter LCD so that it only uses UTC date times
   #crop datasets to start at correct first starting date, but leave room at end in case of duplicates
   start_ind_dataset2 = find(correctedDate, dates[0])[0]
   end_ind_dataset2 = find(correctedDate, dates[-1])[-1]
   correctedRain=correctedRain[start_ind_dataset2: end_ind_dataset2+1]
   correctedTemp= correctedTemp[start_ind_dataset2: end_ind_dataset2+1]
   correctedTime = correctedTime[start_ind_dataset2: end_ind_dataset2+1]
   correctedDate = correctedDate[start_ind_dataset2: end_ind_dataset2+1]
   correctedSLP = correctedSLP[start_ind_dataset2: end_ind_dataset2+1]
   correctedWind = correctedWind[start_ind_dataset2: end_ind_dataset2+1]
   correctedRH= correctedRH[start_ind_dataset2: end_ind_dataset2+1]
   correctedWindDir =correctedWindDir[start_ind_dataset2: end_ind_dataset2+1]
   #Now nan-average repeating values
   correctedSLP_noRepeats=[]; correctedWindDir_noRepeats=[]; correctedRH_noRepeats=[]; correctedWind_noRepeats =[];
   correctedRain_noRepeats=[]; correctedTemp_noRepeats =[]; timeCorrected_noRepeats=[]; i=0; dateCorrected_noRepeats=[]
   while i < len(correctedTime): #i think we need -1 because we forward i in the loop
      #j=0; tmpRain=[];tmpTemp=[]
         j=1; tmpRain, tmpTemp, tmpS, tmpW,tmpWD,tmpRH = [],[],[],[],[],[]
      #try: 
         while i+j < len(correctedTime)-1 and correctedTime[i] == correctedTime[i+j]:
            tmpTemp.append(correctedTemp[i+j]); tmpRH.append(correctedRH[i+j])
            tmpRain.append(correctedRain[i+j]); tmpWD.append(correctedWindDir[i+j])
            tmpS.append(correctedSLP[i+j]); tmpW.append(correctedWind[i+j])
            j=j+1
         if len(tmpRain) == 0 and len(tmpTemp) == 0 and len(tmpS) == 0 and len(tmpW) == 0:
            correctedRain_noRepeats.append(correctedRain[i]); correctedRH_noRepeats.append(correctedRH[i])
            correctedTemp_noRepeats.append(correctedTemp[i]); correctedWindDir_noRepeats.append(correctedWindDir[i])
            correctedSLP_noRepeats.append(correctedSLP[i]);correctedWind_noRepeats.append(correctedWind[i])
         else:
            correctedRain_noRepeats.append(np.nanmean(tmpRain)); correctedRH_noRepeats.append(np.nanmean(tmpRH))
            correctedTemp_noRepeats.append(np.nanmean(tmpTemp)); correctedWindDir_noRepeats.append(np.nanmean(tmpWD))
            correctedSLP_noRepeats.append(np.nanmean(tmpS)); correctedWind_noRepeats.append(np.nanmean(tmpW))
         timeCorrected_noRepeats.append(correctedTime[i])
         dateCorrected_noRepeats.append(correctedDate[i])
         if j == 1 and i<len(correctedTime):
            i=i+1
         if j>1 and i<len(correctedTime):
            i=i+j
     #except:
      #     pass
   #finished if
   if Chatty: print('-> Finished averaging duplicate values in station %s dataset'% str(station))
   #VERY quick check to see if all data is available, if not, flag it for later
   missing_dates=[];missing_hours=[]
   if len(dates) == len(list(set(dateCorrected_noRepeats))):
      if Chatty: print('-> No missing dates at station %s' %(stationList[station],))
   else:
      if Chatty: print('-> Missing dates at %s' %(stationList[station],))
      missing_dates.append(stationList[station])
   #next
   if len(list(set(timeCorrected_noRepeats))) == 24:
      if Chatty: print('-> No missing hours at station %s' %(stationList[station],))
   else:
      if Chatty: print('-> Missing hours at %s' %(stationList[station],))
      missing_hours.append(stationList[station])
   #return
   if Chatty: print('-> Done (%.2f)'%(time.time()-big_start,))
   return correctedRain_noRepeats, correctedTemp_noRepeats, dateCorrected_noRepeats,correctedSLP_noRepeats, correctedWind_noRepeats,  correctedWindDir_noRepeats, correctedRH_noRepeats, timeCorrected_noRepeats, UTC_offset


# ----------------------------------------------------
# Easy bounding box checker
def checkbounds(x,y,x1,y1,x2,y2):
    if (x<x2 and y<y2 and x>x1 and y>y1):
        return True
    else:
        return False

# --# ----------------------------------------------------
def findStations():
   # Get station names -- NOTE1: LCD station names has no header... may cause index errors if format is different!
   stationList=[]; tmp=[]
   listOfStations= pd.read_csv(listOfStationsFile, header=None)
   listOfStations = listOfStations[np.isfinite(listOfStations[5])]   #remove missing station data
   listOfStations =listOfStations.reset_index()   #be able to index the stations properly
   tmp= [format(listOfStations[0][i],'06') for i in range(len(listOfStations))]   #add leading zeroes to stations
   listOfStations['format'] = tmp; del tmp  # add string names to list of stations
   stationList=[str(int(listOfStations[5][i]))+listOfStations['format'][i]+".csv" for i in range(len(listOfStations))]
   stn_lat =listOfStations[15].to_list(); stn_lon =listOfStations[16].to_list()
   stn_latCopy= stn_lat.copy(); stn_lonCopy= stn_lon.copy()
   lenOriginalStations=len(stn_lat)
   stnListCpy = [x for x in stationList]
   in_d01=[]
   #Check bounds and remove from non d01 domains
   for z in range(lenOriginalStations):
      x,y= stn_lat[z],stn_lon[z]
      if checkbounds(x,y,latd02min, lond02min, latd02max, lond02max):
         in_d01.append(True)
      else:
         stnListCpy.remove(stationList[z])
         stn_latCopy.remove(stn_lat[z])
         stn_lonCopy.remove(stn_lon[z])
   #get rid of duplicates
   stationList = [x for x in stnListCpy]
   stn_lat = [x for x in stn_latCopy]
   stn_lon = [x for x in stn_lonCopy]
   del stnListCpy, stn_latCopy, stn_lonCopy
   # [in]Sanity check
   #plt.scatter(stn_lon , stn_lat,c= in_d03); xd03=[lond03min, lond03min, lond03max, lond03max]; yd03=[latd03min, latd03max, latd03min, latd03max]
   #plt.scatter(xd03, yd03) ; plt.show()
   #Check if stations exist and are in domain bounds, if not remove the station
   import requests
   stnListCpy = [x for x in stationList]; stn_latCopy= stn_lat.copy(); stn_lonCopy= stn_lon.copy()
   for station in range(len(stationList)):
      LCD = requests.get(NOAAdataLink + stationList[station])
         #LCD.connect()
      if LCD.status_code > 200:
         if Chatty: print("-> Link does not exist for %s, removing station" %(stationList[station],))
         stnListCpy.remove(stationList[station])
         stn_latCopy.remove(stn_lat[station])
         stn_lonCopy.remove(stn_lon[station])
   #Remove copies again
   stationList = [x for x in stnListCpy]
   stn_lat = [x for x in stn_latCopy]
   stn_lon = [x for x in stn_lonCopy]
   del stnListCpy, stn_latCopy, stn_lonCopy
   # now check to see which of these are within d02, d03 domains
   in_d02=[False for z in range(len(stn_lat))]; in_d03=[False for z in range(len(stn_lat))]
   for z in range(len(stationList)):
      x,y= stn_lat[z],stn_lon[z]
      if (checkbounds(x,y,latd02min, lond02min, latd02max, lond02max)):
          in_d02[z]=True
      if (checkbounds(x,y,latd03min, lond03min, latd03max, lond03max)):
          in_d03[z]=True
   # !!!!!!!!!!----------  !!!!!!!!!!----------- !!!!!!!!!!----------- !!!!!!!!!!
   # to write out station list so we don't need to do this again:
   # !!!!!!!!!!----------- !!!!!!!!!!----------- !!!!!!!!!!----------- !!!!!!!!!! 
   station_out=pd.DataFrame(stationList)
   station_out.columns = ['stn']
   station_out['lat']= stn_lat
   station_out['lon']= stn_lon
   station_out['in_d02']= in_d02
   station_out['in_d03']=in_d03
   return station_out

# remove missing files 
def rm_missing(filenames_d01):
   testrm=[]
   for i in filenames_d01:
      try:
         test=Dataset(dirToWRF+i)
      except FileNotFoundError:
         print(i)
         testrm.append(i)
   # 
   for i in testrm:
       filenames_d01.remove(i)
   #return
   return filenames_d01

#------------------------------------------------------------------------------
# ~~~~~~ START MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------------------------------ load in wrf file names ------------------------
# $1 Get WRF file names
filenames_d02=[]
os.chdir(dirToWRF_d02)
for file in glob.glob("wrfout_d01_*"):
    filenames_d02.append(file)

#filenames_d02=filenames_d02.sort() #files are now sorted by date and time
#print(filenames_d02)
filenames_d02.sort()

# $1 Get WRF file names
filenames_d03=[]
os.chdir(dirToWRF_d03)
for file in glob.glob("wrfout_d01_*"):
    filenames_d03.append(file)

#filenames_d03=filenames_d03.sort() #files are now sorted by date and time
filenames_d03.sort()

#dates=[filenames_d01[z].split("wrfout_d01_")[1].split("_00:00:00")[0] for z in range(len(filenames_d01))]

# pull indices for d0#
#assuming all files with d0# are in the same grid
#remove missing files   
#filenames_d02=rm_missing(filenames_d02)
#filenames_d03=rm_missing(filenames_d03)
dates=[filenames_d02[z].split("wrfout_d01_")[1].split("_00:00:00")[0] for z in range(len(filenames_d02))]
#print(dates)
if month == '01': dates=dates[11:]
print(dates)

#wrf_latd02, wrf_lond02 = latlon_coords(getvar(Dataset(dirToWRF+filenames_d02[1]),"RAINNC"))
#wrf_latd03, wrf_lond03 = latlon_coords(getvar(Dataset(dirToWRF+filenames_d03[1]),"RAINNC"))
wrf_latd02,wrf_lond02 = np.asarray(Dataset(dirToWRF_d02+grid)['LAT']),np.asarray(Dataset(dirToWRF_d02+grid)['LON'])
wrf_latd03,wrf_lond03 = np.asarray(Dataset(dirToWRF_d03+grid)['lat']),np.asarray(Dataset(dirToWRF_d03+grid)['lon'])

#get corners of wrf files
latd02min, latd02max, lond02min, lond02max = wrf_latd02.min().min(), wrf_latd02.max().max(),wrf_lond02.min().min(),wrf_lond02.max().max()


latd03min, latd03max, lond03min, lond03max = wrf_latd03.min().min(), wrf_latd03.max().max(),wrf_lond03.min().min(),wrf_lond03.max().max()


#--------------------------------------------------------------------------
#---------------------- station parsing so we get lat lons ----------------
#------------------------ check to see if you must do this again  ---------

#if written out already
if written:
   station_out=pd.read_csv(dirout+station_out_name)
   stn_lat= station_out['lat']
   stn_lon= station_out['lon']
   stationList =station_out['stn']
   in_d02=  station_out['in_d02']
   in_d03=  station_out['in_d03']
else:
   station_out = findStations()
   station_out.to_csv(dirout+station_out_name)
   station_out=pd.read_csv(dirout+station_out_name)
   stn_lat= station_out['lat']
   stn_lon= station_out['lon']
   stationList =station_out['stn']
   in_d02=  station_out['in_d02']
   in_d03=  station_out['in_d03']

# --------------------------------------------------------------------------------
# ----------------------- get Station dat

# ----------------------- get Station data ---------------------------------------  

# get indices for wrf given stn lat lon
xx_d01=[]; xx_d02=[]; xx_d03=[]; yy_d01=[]; yy_d02=[]; yy_d03=[]
# pull indices for d0#
# assuming all files with d0# are in the same grid
xx_d02,yy_d02=find_index(stn_lon, stn_lat, wrf_lond02, wrf_latd02)
xx_d03,yy_d03=find_index(stn_lon, stn_lat, wrf_lond03, wrf_latd03)

# Start pulling station data to compare
# Output is a list of values for each station

if Chatty: print('-'*70+'\n Starting processing station data \n' + '-'*70)

# Pull out station data ... each rain[box] is a long list of 
rain_real=[[] for t in range(len(yy_d02))]
temp_real=[[] for t in range(len(yy_d02))]
slp_real=[[] for t in range(len(yy_d02))]
RH_real=[[] for t in range(len(yy_d02))]
WindDir_real=[[] for t in range(len(yy_d02))]
Wind_real=[[] for t in range(len(yy_d02))]
Date_real=[[] for t in range(len(yy_d02))]
Time_real=[[] for t in range(len(yy_d02))]
start_out=time.time(); offsets=[]

for station in range(len(yy_d02)):
#for station in range(int(len(yy_d01)/10)):
   start=time.time()
   LCD = pd.read_csv(NOAAdataLink + stationList[station])
   #new loop
   if Chatty: print('-'*70)
   #letemknow
   correctedRain_noRepeats, correctedTemp_noRepeats, dateCorrected_noRepeats,correctedSLP_noRepeats, correctedWind_noRepeats,  correctedWindDir_noRepeats, correctedRH_noRepeats, timeCorrected_noRepeats, UTC_offset = getRealData(LCD)
   #make variable with all station data so we can compare to wrfout
   if Chatty: print('-> Completed loop %s (%s) in %.2f seconds' %(str(station),stationList[station],(time.time()-start)))
   rain_real[station]=correctedRain_noRepeats
   temp_real[station]=correctedTemp_noRepeats
   slp_real[station]=correctedSLP_noRepeats; RH_real[station]=correctedRH_noRepeats; Wind_real[station]=correctedWind_noRepeats
   WindDir_real[station]=correctedWindDir_noRepeats; Time_real[station]=timeCorrected_noRepeats; Date_real[station]=dateCorrected_noRepeats
   #if int(len(yy_d01)/4)==station:
   #    print('25% complete, %.2f' %(time.time()-start_out,))
   #elif int(len(yy_d01)/2)==station:
   #    print('50% complete, %.2f' %(time.time()-start_out,))
   #elif int(3*len(yy_d01)/4)==station:
   #    print('75% complete, %.2f' %(time.time()-start_out,))
   offsets.append(UTC_offset)

xx_d02_list=[xx_d02[i][0] for i in range(len(yy_d02))]; yy_d02_list=[yy_d02[i][0] for i in range(len(yy_d02))]
xx_d03_list=[xx_d03[i][0] for i in range(len(yy_d03))]; yy_d03_list=[yy_d03[i][0] for i in range(len(yy_d03))]


#------------------------- write out data ------------------------------------------------------------
#make header
h=[dateCorrected_noRepeats[i]+'T'+ timeCorrected_noRepeats[i] for i in range(len(timeCorrected_noRepeats))]

#compare station data to wrf station data
writeout_real = pd.DataFrame(temp_real)

#writeout_real.columns = h
writeout_real['xx_d02']= xx_d02_list
writeout_real['yy_d02']= yy_d02_list
writeout_real['xx_d03']= xx_d03_list
writeout_real['yy_d03']= yy_d03_list
writeout_real['lat']=stn_lat
writeout_real['lon']=stn_lon
writeout_real['in_d02']= in_d02
writeout_real['in_d03']= in_d03
writeout_real['UTC_offset']= offsets


writeout_real2= pd.DataFrame(h)

writeout_real.to_csv(dirout+comp_dataset_name)
writeout_real2.to_csv(dirout+comp_dataset_extra)


##compare station data to wrf station data
writeout_real2 = pd.DataFrame(rain_real)

#writeout_real.columns = h
writeout_real2['xx_d02']= xx_d02_list
writeout_real2['yy_d02']= yy_d02_list
writeout_real2['xx_d03']= xx_d03_list
writeout_real2['yy_d03']= yy_d03_list
writeout_real2['lat']=stn_lat
writeout_real2['lon']=stn_lon
writeout_real2['in_d02']= in_d02
writeout_real2['in_d03']= in_d03
writeout_real2['UTC_offset']= offsets

writeout_real2.to_csv(dirout+ comp_dataset_name2)



#I made this loop after writing the script, my bad
zzz= [slp_real, RH_real, WindDir_real, Wind_real]
for i in range(4):
#slp_real=[[] for t in range(len(yy_d01))] RH_real=[[] for t in range(len(yy_d01))]
#WindDir_real=[[] for t in range(len(yy_d01))] Wind_real=[[] for t in range(len(yy_d01))]
   #compare station data to wrf station data
   writeout_real = pd.DataFrame(zzz[i])
   #writeout_real.columns = h
   writeout_real['xx_d02']= xx_d02_list
   writeout_real['yy_d02']= yy_d02_list; writeout_real['xx_d03']= xx_d03_list
   writeout_real['yy_d03']= yy_d03_list; writeout_real['lat']=stn_lat; writeout_real['lon']=stn_lon
   writeout_real['in_d02']= in_d02; writeout_real['in_d03']= in_d03; writeout_real['UTC_offset']= offsets
   writeout_real.to_csv(dirout+comp_dataset_name_list_of4[i])



print("finish step 1")


                               
