
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



dir_WRF='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'
dir_CMAQ = '/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/'
dir_GRID='/projects/b1045/jschnell/ForStacy/latlon_ChicagoLADCO_d03.nc' 
dir_EMIS = '/projects/b1045/wrf-cmaq/input/emis/Chicago_LADCO/ChicagoLADCO_d03/'
emis_dir = '/projects/b1045/wrf-cmaq/input/emis/Chicago_LADCO/ChicagoLADCO_d03/'

#variables of interest
var=['O3','NO2','NO','CO','ISOP','SO2','FORM','PM25_TOT']
wrf_var=['T2','PSFC','RAINC','RAINNC','Q2','V10','U10']
smoke_var = ['NO2','NO','CO','ISOP','SO2', 'FORM']

# User defined functions
#------------------------------------------
def common_data(list1, list2): 
    result = False
    # traverse in the 1st list 
    for x in list1:
        # traverse in the 2nd list 
        for y in list2: 
            # if one common 
            if x == y: 
                result = True
                return result
    return result

#------------------------------------------

#load chicago shapefile
path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)

#get names of files given directoy
onlyfiles = next(os.walk(dir_CMAQ))[2]
onlyfiles=sorted(onlyfiles)
fnames_cmaq = [x for x in onlyfiles if x.startswith("COMBINE_ACONC")]
fnames_wrf= ['wrfout_d01_'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][0:4]+'-'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][4:6]+'-'+str(fnames_cmaq[i]).split('_')[-1].split('.nc')[0][6:]+'_00:00:00' for i in range(len(fnames_cmaq))]

fnames_cmaq = fnames_cmaq[:-1]
fnames_wrf = fnames_wrf[:-1]

#dates
dates=[fnames_wrf[i].split('wrfout_d01_')[1].split('_')[0] for i in range(len(fnames_wrf))]
dates = dates[:-1]
dates2 = ['2018'+'08'+("{:02d}".format(i)) for i in range(1,32)]

version = 'emissions_v0'
# emissions dir
#Get number of files in directory with L2 domain CSV files
emis_files = next(os.walk(emis_dir))[2]
emis_files = [x for x in emis_files if x.startswith("emis_mole_all")]
emis_files =sorted(emis_files) # so that searching for dates are easier
maskfiles = [common_data(emis_files[i].split('_'), dates2) for i in range(len(emis_files))]
emis_files = np.array(emis_files)[maskfiles]


#pull in model files and variables
# for example: finding the difference between the 11th day and the 0th day of NO2:
# cmaq_ncfile[10]['NO2'][0]-cmaq_ncfile[0]['NO2'][0]
cmaq_ncfile= [Dataset(dir_CMAQ+ fnames_cmaq[i],'r') for i in range(len(fnames_cmaq))]
wrf_ncfile=[Dataset(dir_WRF + fnames_wrf[i],'r') for i in range(len(fnames_wrf))]
emis_ncfile=[Dataset(dir_EMIS + emis_files[i],'r') for i in range(len(emis_files))]

units_cmaq = [cmaq_ncfile[0][var[i]].units for i in range(len(var))]
units_wrf = [wrf_ncfile[0][wrf_var[i]].units for i in range(len(wrf_var))]
units_smoke = [emis_ncfile[0][smoke_var[i]].units for i in range(len(smoke_var))]

#get lat lon from grid file
ll=Dataset(dir_GRID,'r')
lat,lon=ll['lat'][:],ll['lon'][:]

#wrflatlon
wrflon, wrflat = wrf_ncfile[0]['XLONG'][0],wrf_ncfile[0]['XLAT'][0]


# pull out variables
#------------------------------------------
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry)[2])

# routine to mask mask over chicago shapefile
mask=np.ones(lon.shape,dtype=bool)
mask[:] = False

for i in range(len(lon)):
    for j in range(len(lon[0])):
       pt = Point(lon[i][j],lat[i][j])
       mask[i][j] =  pt.within(union[0])

# routine to mask mask over chicago shapefile
mask_wrf=np.ones(wrflon.shape,dtype=bool)
mask_wrf[:] = False

for i in range(len(wrflon)):
    for j in range(len(wrflon[0])):
       pt = Point(wrflon[i][j], wrflat[i][j])
       mask_wrf[i][j] =  pt.within(union[0])



hours = pd.date_range(dates[0]+" 00:00", dates[-2]+" 23:00",freq="60min")


# pull out variables
#------------------------------------------

def pull_vars(ncfile,var,mask):
   var_crop = []
   for i in range(len(var)):
      if ncfile == cmaq_ncfile: crop = [ncfile[j][var[i]][h][0][mask] for h in range(24) for j in range(len(ncfile))]
      elif ncfile == wrf_ncfile: crop = [ncfile[j][var[i]][h][mask_wrf] for h in range(24) for j in range(len(ncfile))]
      elif ncfile == emis_ncfile: crop = [ncfile[j][var[i]][h][0][mask] for h in range(24) for j in range(len(ncfile))]
      else: print('ERROR')
      #
      var_crop.append(crop)
   return var_crop

mask_ravel = np.array(mask).ravel()
lon_ravel = np.array(lon).ravel()[np.array(mask).ravel()]
lat_ravel = np.array(lat).ravel()[np.array(mask).ravel()]

var_crop=pull_vars(cmaq_ncfile,var,mask)
var_crop_emis=pull_vars(emis_ncfile,smoke_var,mask)
var_crop_wrf=pull_vars(wrf_ncfile,wrf_var,mask_wrf)

#rainc,rainnc = np.asarray(var_crop_wrf[3]), np.asarray(var_crop_wrf[2])
rain_cumulative = np.asarray(var_crop_wrf[3]) + np.asarray(var_crop_wrf[2])

rain = [[] for i in range(len(rain_cumulative))]

# remove the cumulative nature of rain variables
for i in range(len(rain_cumulative)):
  if i == 0: rain[0] = np.zeros(rain_cumulative[0].shape).tolist()
  else: rain[i] = (rain_cumulative[i]-rain_cumulative[i-1]).tolist()

#wrf_var=['T2','PSFC','RAINC','RAINNC','Q2','V10','U10']
var_crop_wrf = np.array([var_crop_wrf[0]]+ [var_crop_wrf[1]]+ [var_crop_wrf[4]]+ [var_crop_wrf[5]]+[var_crop_wrf[6]]+[rain])
wrf_var = ['T2','PSFC','Q2','V10','U10','RAIN']

var_crop_emis_tot = [np.array(var_crop_emis[i]).ravel() for i in range(len(var_crop_emis))]
var_crop_tot = [np.array(var_crop[i]).ravel() for i in range(len(var_crop))]
var_crop_wrf_tot = [var_crop_wrf[i].ravel() for i in range(len(var_crop_wrf))]

var_crop_wrf_tot = var_crop_wrf_tot+ np.array([(np.array(var_crop_wrf_tot[-1])**2+np.array(var_crop_wrf_tot[-2])**2)**.5]).tolist()
wrf_var = wrf_var + ['Wind_TOT']

var_to_wrf = var_crop_tot + var_crop_wrf_tot
var_to_emis = var_crop_tot + var_crop_emis_tot 

#make corr matric
corr_matrix_vw = np.zeros([len(var_to_wrf), len(var_to_wrf)]); corr_matrix_ve = np.zeros([len(var_to_emis), len(var_to_emis)])

from scipy.stats import pearsonr

for i in range(len(var_to_wrf)):
   for j in range(len(var_to_wrf)):
      corr_matrix_vw[i][j]= pearsonr(var_to_wrf[i], var_to_wrf[j])[0]

for i in range(len(var_to_emis)):
   for j in range(len(var_to_emis)):
      corr_matrix_ve[i][j]= pearsonr(var_to_emis[i], var_to_emis[j])[0]

maskvw = np.zeros_like(corr_matrix_vw_df)
maskvw[np.triu_indices_from(maskvw)] = True

maskve = np.zeros_like(corr_matrix_ve_df)
maskve[np.triu_indices_from(maskve)] = True

# Start plotting cmaq v wrf
titles_vw = [var[i]+'_CMAQ' for i in range(len(var))] + [wrf_var[i]+'_WRF' for i in range(len(wrf_var))] 

corr_matrix_vw_df = pd.DataFrame(corr_matrix_vw)
corr_matrix_ve_df = pd.DataFrame(corr_matrix_ve)
corr_matrix_vw_df.columns = titles_vw
corr_matrix_vw_df.index = titles_vw
corr_matrix_ve_df.columns = titles_ve
corr_matrix_ve_df.index = titles_ve

# Make heat maps of variables
fig = plt.subplots(figsize = (8,7))
sns.heatmap(corr_matrix_vw_df,center = 0,annot = True,mask = maskvw, fmt='.2f')
plt.tight_layout()
plt.savefig('correlation_matrix_vw.svg')
plt.show()

# Start plotting cmaq v wrf
titles_ve = [var[i]+'_CMAQ' for i in range(len(var))] + [smoke_var[i]+'_SMK' for i in range(len(smoke_var))] 
corr_matrix_ve_df = pd.DataFrame(corr_matrix_ve)
corr_matrix_ve_df.columns = titles_ve
corr_matrix_ve_df.index = titles_ve

# Make heat maps of variables
fig = plt.subplots(figsize = (8,7))
sns.heatmap(corr_matrix_ve_df,center = 0,annot = True,mask = maskve, fmt='.2f')
plt.tight_layout()
plt.savefig('correlation_matrix_ve.svg')
plt.show()
