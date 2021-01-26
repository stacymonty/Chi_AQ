#!/usr/bin/env python

# -----------------
# Step 2
# -----------------
# from step 1, get indices of the LCD latlon, which is used as an input to this code.
# This code pulls out WRF data into csv files in the order of the LCD station data, which is then used as input to code 3. 

#ERROR-- rain seems to be weird. check write out. plot rain variables

# ---------------------------------------------------------------------------------------------------------
# ~~~~~~ START USER INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# variables of interest
minTemp = 242; maxTemp = 294;
month='08'
year='2018'
# Location of WRF output
runname='output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852'
dirout='/home/asm0384/WRFcheck/'+runname+'/'

# Processed US data, from previous file
#File out names
comp_dataset_name = dirout+'wrfcheck_withstations_'+runname+'_'+month+year+'.csv'                     # name and directory to write out to
comp_dataset_extra = dirout+'completeddata_mini_extras2.csv'
station_out_name = dirout+'station_out_removedmissing.csv' #name of intermediate file
comp_dataset_name2= dirout+'wrfcheck_withstations_complete_rain.csv'



#location of wrf and filenames
#dirToWRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/'+runname+'/'
# Name of run
runname='output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852'
#BASE_PXLSM_v0
# Location of WRF output

dirToWRF_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'
dirToWRF_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/'
listOfStationsFile = "~/lcd-stations.csv"
dirout='/home/asm0384/WRFcheck/'+runname+'/'
grid='latlon.nc'


#dirout='/home/asm0384/WRFcheck/'+runname+'/'
Chatty= True       # false if you want to remove print statements
slpon= True #True  #need to configure to make SLP

#start the code
if Chatty: print('Starting ....')

# --------------------------------------------------------------------------------------------------------
def getWRFfromIND(ncfile,indxy, filenames,varname):
    t2d01=[ncfile[z][varname][i] for z in range(len(ncfile)) for i in range(24)]
    t2d01_xx=  [[t2d01[t][indxy[l]] for t in range(24*len(ncfile))] for l in range(len(indxy))]
    return t2d01_xx

def getslpfromIND(ncfile,indxy, filenames,varname):
    t2d01=[ncfile[z][varname][i] for i in range(24) for z in range(len(ncfile))]
    t2d01_xx= [[t2d01[t][indxy[l]] for t in range(24*len(ncfile))] for l in range(len(indxy))]
    return t2d01_xx

def getRHfromIND(ncfile,indxy, filenames):
    pq0 = 379.90516; a2 = 17.2693882; a3 = 273.16; a4 = 35.86
    q2=[ncfile[z]['Q2'][i]/((pq0 / ncfile[z]['PSFC'][i]) **(a2 * (ncfile[z]['T2'][i] - a3) / (ncfile[z]['T2'][i] - a4))) for z in range(len(ncfile)) for i in range(24)]
    t2d01_xx=  [[q2[t][indxy[l]] for t in range(24*len(ncfile))] for l in range(len(indxy))]
    return t2d01_xx

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

#t2d01=[getvar(ncfiled01[z],"slp",timeidx=i).data for i in range(24) for z in range(len(filenames_d01))]

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


#------------------------------------------------------------------------------
# ~~~~~~ START MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------------------------------ load in wrf file names ------------------------
# $1 Get WRF file names
#filenames_d01=[] 
#os.chdir(dirToWRF)
#for file in glob.glob("wrfout_d01_*"):
#    filenames_d01.append(file)
#
#filenames_d01.sort() #files are now sorted by date and time

# $1 Get WRF file names
filenames_d02=[]
os.chdir(dirToWRF_d02)
for file in glob.glob("wrfout_d01_*"):
    filenames_d02.append(file)

filenames_d02.sort() #files are now sorted by date and time
filenames_d02=filenames_d02[:-1]

# $1 Get WRF file names
filenames_d03=[]
os.chdir(dirToWRF_d03)
for file in glob.glob("wrfout_d01_*"):
    filenames_d03.append(file)

filenames_d03.sort() #files are now sorted by date and time
filenames_d03=filenames_d03[:-1]

# remove missing files
#filenames_d01=rm_missing(filenames_d01)
#filenames_d02=rm_missing(filenames_d02)
#filenames_d03=rm_missing(filenames_d03)

#dates_d01=[filenames_d01[z].split("wrfout_d01_")[1].split("_00:00:00")[0] for z in range(len(filenames_d01))]
dates_d02=[filenames_d02[z].split("wrfout_d01_")[1].split("_00:00:00")[0] for z in range(len(filenames_d02))]
dates_d03=[filenames_d03[z].split("wrfout_d01_")[1].split("_00:00:00")[0] for z in range(len(filenames_d03))]

#if dates_d01== dates_d02 and d
ates=dates_d01
#else:
#   print('dates are not consistent between domains! Defaulting to d01 dates, may cause errors!')
dates=dates_d02

#ncfiled01 = [Dataset(filenames_d01[i]) for i in range(len(filenames_d01))]
ncfiled02 = [Dataset(dirToWRF_d02+filenames_d02[i]) for i in range(len(filenames_d02))]
ncfiled03 = [Dataset(dirToWRF_d03+filenames_d03[i]) for i in range(len(filenames_d03))]


#get indices for dataset, compress  the indices for each domain
STATION= pd.read_csv(comp_dataset_name)
in_d02= STATION['in_d02'].tolist()
in_d03 = STATION['in_d03'].tolist()
yy_d02=np.compress(in_d02,STATION['yy_d02']).tolist();xx_d02= np.compress(in_d02, STATION['xx_d02']).tolist()
yy_d03= np.compress(in_d03, STATION['yy_d03']).tolist();xx_d03= np.compress(in_d03, STATION['xx_d03']).tolist()

indxyd02clip =[(xx_d02[t],yy_d02[t]) for t in range(len(yy_d02))]
indxyd03clip =[(xx_d03[t],yy_d03[t]) for t in range(len(yy_d03))]
print(indxyd02clip)
#pull variables
start=time.time()
#t2d01 = getWRFfroimIND(ncfiled01,indxyd01, filenames_d01,'T2')
t2d02 = getWRFfromIND(ncfiled02, indxyd02clip, filenames_d02,'T2')
t2d03 = getWRFfromIND(ncfiled03, indxyd03clip, filenames_d03,'T2')

#raind01 = getWRFfromIND(ncfiled01,indxyd01, filenames_d01,'RAINC')
raind02 = getWRFfromIND(ncfiled02, indxyd02clip, filenames_d02,'RAINC')
raind03 = getWRFfromIND(ncfiled03, indxyd03clip, filenames_d03,'RAINC')

#rainncd01 = getWRFfromIND(ncfiled01,indxyd01, filenames_d01,'RAINNC')
rainncd02 = getWRFfromIND(ncfiled02, indxyd02clip, filenames_d02,'RAINNC')
rainncd03 = getWRFfromIND(ncfiled03, indxyd03clip, filenames_d03,'RAINNC')

#rhd01 = getRHfromIND(ncfiled01,indxyd01, filenames_d01)
rhd02 = getRHfromIND(ncfiled02, indxyd02clip, filenames_d02)
rhd03 = getRHfromIND(ncfiled03, indxyd03clip, filenames_d03)

# 10 might be wrong

# 10 might be wrong
#u10d01,v10d01  = getWRFfromIND(ncfiled01,indxyd01, filenames_d01,'U10'),getWRFfromIND(ncfiled01,indxyd01, filenames_d01,'V10')
u10d02,v10d02 =getWRFfromIND(ncfiled02,indxyd02clip,filenames_d02,'U10'),getWRFfromIND(ncfiled02,indxyd02clip,filenames_d02, 'V10')
u10d03,v10d03 = getWRFfromIND(ncfiled03,indxyd03clip, filenames_d03, 'U10'),getWRFfromIND(ncfiled03,indxyd03clip, filenames_d03, 'V10')


if slpon==True:
#   slpd01 = getslpfromIND(ncfiled01,indxyd01, filenames_d01,'PSFC')
   slpd02 = getslpfromIND(ncfiled02, indxyd02clip, filenames_d02,'PSFC')
   slpd03 = getslpfromIND(ncfiled03, indxyd03clip, filenames_d03,'PSFC')

end=str(time.time()-start)
print('Time to pull variables from netCDF files: '+ end + 's')


#q=[t2d01, t2d02, t2d03, raind01, raind02, raind03, rainncd01, rainncd02, rainncd03,rhd01,rhd02,rhd03, u10d01,v10d01,u10d02,v10d02,u10d03,v10d03]
#q1=['t2d01', 't2d02', 't2d03', 'raind01', 'raind02', 'raind03', 'rainncd01', 'rainncd02', 'rainncd03']
#del t2d01, t2d02, t2d03, raind01, raind02, raind03, rainncd01, rainncd02, rainncd03
q=[t2d02, t2d03,raind02, raind03, rainncd02, rainncd03, rhd02,rhd03,u10d02,u10d03,v10d02,v10d02]
#name=['t2d01.csv', 't2d02.csv', 't2d03.csv', 'raind01.csv', 'raind02.csv', 'raind03.csv', 'rainncd01.csv', 'rainncd02.csv', 'rainncd03.csv',]
#name=['t2d01.csv', 't2d02.csv', 't2d03.csv', 'raind01.csv', 'raind02.csv', 'raind03.csv', 'rainncd01.csv', 'rainncd02.csv', 'rainncd03.csv','rhd01.csv','rhd02.csv','rhd03.csv', 'u10d01.csv','v10d01.csv','u10d02.csv','v10d02.csv','u10d03.csv','v10d03.csv']
name=['t2d02.csv', 't2d03.csv',  'raind02.csv', 'raind03.csv', 'rainncd02.csv', 'rainncd03.csv','rhd02.csv','rhd03.csv', 'u10d02.csv','u10d03.csv', 'v10d02.csv','v10d03.csv']

for i in range(len(q)):
   df= pd.DataFrame(q[i])
   df.to_csv(dirout+name[i])


if slpon==True:
   #q1=[slpd01, slpd02, slpd03]
   #name1=['slpd01.csv', 'slpd02.csv', 'slpd03.csv']
   q1=[slpd02, slpd03]
   name1=[ 'slpd02.csv', 'slpd03.csv']
   for i in range(len(q1)):
      df= pd.DataFrame(q1[i])
      df.to_csv(dirout+name1[i])


print("Done with step 2")


