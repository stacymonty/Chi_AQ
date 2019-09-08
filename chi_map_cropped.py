#------------------------------------------
# Fun figure for AGU CVD
# Stacy Montgomery, Sept. 2019
#
# I made some AQ figures, the interesting part is "mpath" 
# and using the "exterior" of the shapefile to crop the figure
#------------------------------------------

#------------------------------------------
# Libraries
#--------------
from matplotlib import pyplot as plt
from mpl_toolkits import basemap as bm
from matplotlib import colors
import numpy as np
import numpy.ma as ma
from matplotlib.patches import Path, PathPatch
import pandas as pd
from shapely.geometry import Point, shape, Polygon
import fiona
from shapely.ops import unary_union, cascaded_union
from geopandas.tools import sjoin
import geopandas as gpd
import geoplot
import glob
import os
from datetime import timedelta, date;
from netCDF4 import Dataset
import scipy.ndimage
from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
import matplotlib.path as mpath
import seaborn as sns

#------------------------------------------
# Find index of points on a gridded array
# stn_lon,stn_lat = list of lat lon points --> lat_list, lon_list = [x1,x2][y1,y2]
# wrf_lon, wrf_lat = np.array of gridded lat lon --> grid_x= np.array([x1,x2,x3],[x4,x5,x6])
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

#------------------------------------------
# USER INPUT
fout_dir_l3='/home/asm0384/tempfiles/practice/NO2_l3_big/'
plot_file='L3_averaged_Chicago_L2_Chicago_2018-8-1_through_L2_Chicago_2018-8-30_made_1567447847_nx_1250_ny_1000.csv'
varname= 'no2'
path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'

# Start pulling and cropping data
chi  = gpd.GeoDataFrame.from_file(path)

#data frame with all data
finalgrid = pd.read_csv(fout_dir_l3+plot_file, index_col =0)
varname ='nitrogendioxide_tropospheric_column'

#Pull information from title
filename= plot_file
ymd= plot_file.split('_made_')
nxny=ymd[1].split('_nx_')[1].split('_ny_')
nx=int(nxny[0])
ny=int(nxny[1].split('.csv')[0])
startdate=ymd[0].split('L3_averaged_Chicago_L2_Chicago_')[1].split('_through')[0]
enddate=ymd[0].split('L3_averaged_Chicago_L2_Chicago_')[1].split('_through_L2_Chicago_')[1]

finalgrid.describe()

# NOW CROP OVER CHICAGO
# Initialize grid
grid_nlat=np.zeros((ny,nx)); grid_nlon=np.zeros((ny,nx)); grid_no2=np.zeros((ny,nx))

# Return back to grid form
for i in range(ny):
    for j in range(nx):
        l=i*nx+j
        grid_nlat[i][j]=finalgrid['nlats'][l]
        grid_nlon[i][j]=finalgrid['nlons'][l]
        grid_no2[i][j]=finalgrid[varname][l]

# Check 
#plt.scatter(finalgrid['nlons'],finalgrid['nlats'],c=finalgrid['nitrogendioxide_tropospheric_column'])
#plt.show()

# Make box around chicago to cut data -- specific for satellite, check to make sure the arrays are increasing in size
x1,y1=find_index([min(chi.bounds.minx)],[min(chi.bounds.miny)], np.array(grid_nlon), np.array (grid_nlat))
x2,y2=find_index([max(chi.bounds.maxx)],[max(chi.bounds.maxy+.05)], np.array(grid_nlon), np.array (grid_nlat))
x3,y3=find_index([min(chi.bounds.minx)],[max(chi.bounds.maxy)], np.array(grid_nlon), np.array (grid_nlat))
x4,y4=find_index([max(chi.bounds.maxx)+.05],[min(chi.bounds.miny)], np.array(grid_nlon), np.array (grid_nlat))

#set up zeros array given the bound of chicago
diffy =max(y1,y2,y3,y4)[0]-min(y1,y2,y3,y4)[0]
diffx=max(x1,x2,x3,x4)[0]-min(x1,x2,x3,x4)[0]

zlon,zlat,z=np.zeros([diffx, diffy]), np.zeros([diffx, diffy]), np.zeros([diffx, diffy])

# fill out zeros array from the gridded data
for i in range(diffx):
   for j in range(diffy):
      z[i][j]= grid_no2[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j] 
      zlat[i][j]= grid_nlat[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j]
      zlon[i][j]= grid_nlon[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j]

# Check 
#ax= chi.plot()
#plt.scatter(zlon,zlat,c=z)

#plt.show()

# Make the contour plot
# make finer
import scipy.ndimage

from cartopy import crs as ccrs
from cartopy.io.shapereader import Reader
import matplotlib.path as mpath
import seaborn as sns

crs_new = ccrs. AlbersEqualArea(central_longitude=(chi.bounds.mean().minx+chi.bounds.mean().maxx)/2)

#get data at higher resolution for contouring
lat,lon,data=scipy.ndimage.zoom(zlat, 3),scipy.ndimage.zoom(zlon, 3),scipy.ndimage.zoom(z, 3)
data=data*10e4

# merge polygons using unary union and get the outside values
# NOTE -- the union makes a multipolygon, but if you reference the largest of the polygons you actually get the outside
union=gpd.GeoSeries(unary_union(chi.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])

# make fig object
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(5, 5))

#set up data for plotting via levels
vmax=pd.DataFrame(data).max().max()+1.5
vmin= int(pd.DataFrame(data).min().min())+2
levels = np.linspace(vmin, int(vmax), int(vmax)+10)

#locate outside
#plt.scatter(list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1])

# set boundary as outer extent by making a matplotlib path object and adding that geometry
# i think setting the boundary before you plot the data actually crops the data to the shape, so set ax first
axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')

#plot the gridded data by using contourf
cs=plt.contourf(lon,lat,data,cmap= "inferno_r", transform=crs_new, levels=levels)

# add landmarks with scatterplot
midway= 41.7868, -87.7522
ohare = 41.9742, -87.9073
loop = 41.8786, -87.6251
plt.scatter(pd.DataFrame([midway,ohare,loop])[1],pd.DataFrame([midway,ohare,loop])[0],marker = '*',color='white')

# set axes extents from shapefile
x=[min(chi.bounds.minx), max(chi.bounds.maxx)] 
y=[min(chi.bounds.miny), max(chi.bounds.maxy)]
axs.set_extent([x[0]-.03,x[1]+.03,y[0]-.03,y[1]+.03],crs= crs_new)
axs.set_title('1 PM TropOMI NO$_{2}$ Column Density')

#add colorbar and label
cbar=plt.colorbar(cs,boundaries=np.arange(vmin,11))
cbar.ax.set_ylabel('10$^{-2}$ molecules m$^{2}$')
cbar.set_ticks(np.arange(vmin, int(vmax),1))

# save and show
plt.savefig('/home/asm0384/tropomi_no2_neighbs_1_star.pdf',format='pdf')
plt.show()


#------------------------------------------------------------------------------------
# CMAQ Processing
#--------------------------------

print('-----------')
print('Starting CMAQ PROCESSING....')
print('-----------')

# Directories for cmaq + EPA 
dir_cmaq='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_BC_4km_v0/postprocess/'

# CMAQ things
domain='d02'
time='hourly'
year='2018'
#epa_code=['42401','42602','44201']; var=['SO2','NO2','O3'] #numerical identifiers and corresponding vars
epa_code=['42602']; var=['NO2']

# Get CMAQ file names
cmaq_files=[]
os.chdir(dir_cmaq)
for file in glob.glob("COMBINE_ACONC_*"):
    cmaq_files.append(file)

# Find dates from cmaq
cmaq_files.sort();
cmaq_files=cmaq_files[0:-2] #get rid of september

dates=[cmaq_files[z].split("COMBINE_ACONC_")[1].split(".nc")[0] for z in range(len(cmaq_files))]
start_dt=date(int(dates[0][0:4]),int(dates[0][4:6]),int(dates[0][6:8]))
end_dt=date(int(dates[-1][0:4]),int(dates[-1][4:6]),int(dates[-1][6:8]))

#pull data
cmaq=[Dataset(dir_cmaq+cmaq_files[i]) for i in range(len(cmaq_files))]
t_index = pd.DatetimeIndex(start=start_dt, end=end_dt, freq='1h')
dates_ft=[str(date(int(dates[i][0:4]),int(dates[i][4:6]),int(dates[i][6:8]))) for i in range(len(dates))]

#get monthly avg of CMAQ data
monthly_avg_no2=[cmaq[i]['NO2'][h] for i in range(len(cmaq)) for h in range(24)]
monthly_avg_no2= sum(monthly_avg_no2)/(len(cmaq)*24)

# get 1 pm avg no2
pm_avg_no2=[cmaq[i]['NO2'][13] for i in range(len(cmaq))]
pm_avg_no2=sum(monthly_avg_no2)/(len(cmaq))

monthly_avg_no2= pm_avg_no2 #stupid

# get 1 pm avg o3
#monthly_avg_o3=[cmaq[i]['O3'][h] for i in range(len(cmaq)) for h in range(24)]
#monthly_avg_o3= sum(monthly_avg_no2)/(len(cmaq)*24)

#Pull cmaq grid
grid='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d02/GRIDCRO2D_Chicago_LADCO_2018-08-20.nc'
cmaq_lat,cmaq_lon=Dataset(grid)['LAT'][0][0],Dataset(grid)['LON'][0][0]

#check the extent and that everything looks right
#plt.scatter(cmaq_lon, cmaq_lat, c= monthly_avg_no2[0])

# Find indices of the greatest outside points of the data
x1,y1=find_index([min(chi.bounds.minx)],[min(chi.bounds.miny)], np.array(cmaq_lon), np.array (cmaq_lat))
x2,y2=find_index([max(chi.bounds.maxx)],[max(chi.bounds.maxy)], np.array(cmaq_lon), np.array (cmaq_lat))
x3,y3=find_index([min(chi.bounds.minx)],[max(chi.bounds.maxy)], np.array(cmaq_lon), np.array (cmaq_lat))
x4,y4=find_index([max(chi.bounds.maxx)+.05],[min(chi.bounds.miny)], np.array(cmaq_lon), np.array (cmaq_lat))

# make an array the size of the bounds of the shapefile
diffy =max(y1,y2,y3,y4)[0]-min(y1,y2,y3,y4)[0]
diffx=max(x1,x2,x3,x4)[0]-min(x1,x2,x3,x4)[0]
zlon,zlat,z=np.zeros([diffx, diffy]), np.zeros([diffx, diffy]), np.zeros([diffx, diffy])

for i in range(diffx):
   for j in range(diffy):
      z[i][j]= monthly_avg_no2[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j] 
      zlat[i][j]= cmaq_lat[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j]
      zlon[i][j]= cmaq_lon[min(x1,x2,x3,x4)[0]+i][min(y1,y2,y3,y4)[0]+j]

# Make Contour plot
# make finer


crs_new = ccrs. AlbersEqualArea(central_longitude=(chi.bounds.mean().minx+chi.bounds.mean().maxx)/2)

#get data at higher resolution for contouring
lat,lon,data=scipy.ndimage.zoom(zlat, 3),scipy.ndimage.zoom(zlon, 3),scipy.ndimage.zoom(z, 3)

#merge polygons and get the outside valules
b=gpd.GeoSeries(unary_union(chi.geometry))
v=pd.DataFrame([list(b[0][2].exterior.xy)[0], list(b[0][2].exterior.xy)[1]])

# make fig object
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},
                        figsize=(5, 5))

#set up data for plotting via levels
vmax=pd.DataFrame(data).max().max()
vmin= pd.DataFrame(data).min().min()-.007
vmin=.3
levels = np.arange(vmin, vmax+.1, .10)

#locate outside
#plt.scatter(list(b[0][2].exterior.xy)[0], list(b[0][2].exterior.xy)[1])

#set boundary as outer extent
axs.set_boundary(mpath.Path(v.T,closed=True), transform= crs_new, use_as_clip_path=True)

axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
cs=plt.contourf(lon,lat,data,cmap= "inferno_r", transform=crs_new, levels=levels)

x=[min(chi.bounds.minx), max(chi.bounds.maxx)] 
y=[min(chi.bounds.miny), max(chi.bounds.maxy)]

axs.set_extent([x[0]-.03,x[1]+.03,y[0]-.03,y[1]+.03],crs= crs_new)
axs.set_title('1 PM CMAQ NO$_{2}$ Ground Level')

cbar=plt.colorbar(cs,boundaries=np.arange(vmin,11))
cbar.ax.set_ylabel('ppbV')
cbar.set_ticks(np.arange(vmin, vmax,.2))

midway= 41.7868, -87.7522
ohare = 41.9742, -87.9073
loop = 41.8786, -87.6251

plt.scatter(pd.DataFrame([midway,ohare,loop])[1],pd.DataFrame([midway,ohare,loop])[0],marker = '*',color='white')

plt.savefig('/home/asm0384/cmaq_no2_neighbs_1pm.pdf',format='pdf')

plt.show()


#------------------------------------------------------------------------------------
# INCOME Processing
# This could be done better, I will in the future sort my own chloropleth, the geoplot funciton is
# not broad enough, but it's quick enough to work for me
#--------------------------------
import geoplot.crs as gcrs

fname='/home/asm0384/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv'
income=pd.read_csv(fname)

path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'

chi  = gpd.GeoDataFrame.from_file(path)

income['COMMUNITY AREA NAME'][75]="OHARE"

income['community']=[income['COMMUNITY AREA NAME'][i].upper() for i in range(len(income))]


dfmerge=pd.merge(chi,income,on='community')

# dropna cloropleth graph ... 
c=geoplot.choropleth(dfmerge, hue = dfmerge['HARDSHIP INDEX'],
    cmap='Blues', figsize=(5, 5), k=None, legend=True)
    #legend_values=np.arange(10000,90000,10000))


plt.title('Hardship Index')

# scatter landmarks
midway= 41.7868, -87.7522
ohare = 41.9742, -87.9073
loop = 41.8786, -87.6251

#oops doesnt work ... add in post processing ...
plt.scatter(pd.DataFrame([midway,ohare,loop])[1],pd.DataFrame([midway,ohare,loop])[0],marker = '*',color='white')

plt.savefig('/home/asm0384/HI_neighbs_hot_1.5.pdf',format='pdf')

plt.show()
