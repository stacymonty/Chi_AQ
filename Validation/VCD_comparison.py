#!/usr/bin/env python

# ---------------------------------------------------------------------
# Stacy Montgomery, Jan 2021
#
# Use after you crop the L2 

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

# Projections -- this will be used in naming files later
domain = 'Chicago'
# grid file
grid='/home/asm0384/ChicagoStudy/inputs/grid/latlon_ChicagoLADCO_d03.nc'
lon,lat = np.array(Dataset(grid,'r')['lon']),np.array(Dataset(grid,'r')['lat'])

var='NO2'

#Directory to where L2 TropOMI files are stored
dir='/projects/b1045/TropOMI/'+var+'/l2_cut/'

#from netcdf file, what do you want
varname='nitrogendioxide_tropospheric_column'
varprecision='qa_value'
tagdir = '~/tag/'

filestartswith  = 'S5P_OFFL_L2__NO2____' # 'S5P_OFFL_L2__O3'

summer_regrid = pd.read_csv('~/rbdinterp_linear_smooth_201808.csv',index_col=0)
summer_regrid2 = pd.read_csv('~/rbdinterp_linear_smooth_201808_pt2.csv',index_col=0)

wint_avg_trop= np.asarray(pd.read_csv('~/rbdinterp_linear_smooth_201901_NO2.csv',index_col=0))*1000
summer_avg_trop = np.asarray((summer_regrid2+summer_regrid)/2)*1000


# pull in column

dwint = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/'
dsum = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'

filestartswith  = 'CCTM_CONC_v385' # 
fs = next(os.walk(dwint))[2]
fs = [x for x in fs if x.startswith(filestartswith)]
f_wint=sorted(fs)
fs = next(os.walk(dsum))[2]
fs = [x for x in fs if x.startswith(filestartswith)]
f_sum=sorted(fs)

time = 13
summer_cmaq_trop = np.asarray([Dataset(dsum+f_sum[i])['NO2'][time][0:31].sum(axis=0) for i in range(len(f_sum))])
winter_cmaq_trop = np.asarray([Dataset(dwint+f_wint[i])['NO2'][time][0:31].sum(axis=0) for i in range(len(f_wint))])

summer_cmaq_avg_trop = summer_cmaq_trop.mean(axis=0)
winter_cmaq_avg_trop =winter_cmaq_trop.mean(axis=0)



# functions
def stats_normalized(data,prediction):
	x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
	mu_d,mu_p = np.mean(x),np.mean(y)
	nmb = np.sum(y-x)/np.sum(x)*100
	nme = np.sum(np.abs(y-x))/np.sum(x)*100
	r,p = st.pearsonr(x,y)
	return mu_d,mu_p,nmb,nme,r,p

tropstats = pd.DataFrame(np.asarray([stats_normalized(summer_avg_trop,summer_cmaq_avg_trop),stats_normalized(wint_avg_trop,winter_cmaq_avg_trop)]))
tropstats.columns = ['mu_d','mu_p','bias','rmse','r','p']
tropstats.index = ['Summer 2018','Winter 2019']
tropstats.to_csv('~/TroposphereStats.csv')

