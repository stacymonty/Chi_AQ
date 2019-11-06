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

# shapes and directories
# shapefile == https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=State+Legislative+Districts
path='/home/asm0384/shapefiles/replines/tl_2019_17_sldl.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
rep_districts_shapefile = gpd.GeoDataFrame.from_file('/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp')

# dir to grid file
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 

# dir to model files
dir_files='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_1.33km_sf_rrtmg_5_10_1_v3852/'

# ---------------------------------------------------------------------
# START
# ---------------------------------------------------------------------

#pull files from given directoy
onlyfiles = next(os.walk(dir_files))[2]
onlyfiles=sorted(onlyfiles) # so that searching for dates are easier

# pull only CONC files
fnames = [x for x in onlyfiles if x.startswith("CCTM_CONC_")]
numfiles=(len(fnames))

# Days and months we're interested in:
datesofinterest=np.arange(startday,endday+1)
monthsofinterest=np.arange(startmonth,endmonth+1)

#get lat lon from grid file
ll=Dataset(dir+ll,'r')
lat,lon=ll['lat'][:],ll['lon'][:]

#pull in files and variables
ncfile= [Dataset(dir_files+fnames[i],'r') for i in range(len(fnames))]

#full day conc
no2 = [np.average(ncfile[i]['NO2'][:],axis=0) for i in range(len(fnames))]
#hourly conc
no2= [ncfile[i]['NO2'][13] for i in range(len(fnames))]
no2_hourly=np.average(no2,axis=0)

O3 = [np.average(ncfile[i]['O3'][:],axis=0) for i in range(len(fnames))]

O3_hourly = [ncfile[i]['O3'][18] for i in range(len(fnames))]
O3=np.average(O3,axis=0)

CO = [np.average(ncfile[i]['CO'][:],axis=0) for i in range(len(fnames))]
CO=np.average(CO,axis=0)

# get outside shape of shapefile to do plotting
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0].exterior.xy)[0], list(union[0].exterior.xy)[1]])

#==================================================

# set var for plot
var='NO2'
data=pd.DataFrame(no2[0]*10e2)

#==================================================

# files
crs_new = ccrs.AlbersEqualArea(central_longitude=(chi_shapefile.bounds.mean().minx+chi_shapefile.bounds.mean().maxx)/2)

# make fig object
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 6))

#set up data for plotting via levels
vmax=int(pd.DataFrame(data).max().max())-30
vmin= int(pd.DataFrame(data).min().min())
levels = np.linspace(vmin, vmax, 20)

#vmin,vmax= 15,130
#levels = np.linspace(vmin, vmax, 20)

# get rid of values outside the levels we are contouring to
data[pd.DataFrame(data)>vmax]=vmax

# set boundary as outer extent by making a matplotlib path object and adding that geometry
# i think setting the boundary before you plot the data actually crops the data to the shape, so set ax first
axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
#bold dis 45
axs.add_geometries(gpd.geoseries.GeoSeries(chi_shapefile[chi_shapefile['NAMELSAD']=='State House District 45'].geometry), crs=crs_new,facecolor='none', edgecolor='black', linewidth=2.0)

#plot the gridded data by using contourf
cs=plt.contourf(lon,lat,data,cmap= "inferno", transform=crs_new, levels=levels)
# add landmarks with scatterplot
midway=  -87.7522,41.7868
ohare = -87.9073, 41.9842
loop =  -87.6251,41.8786
axs.annotate(xy=midway,s="Midway",color='white')
axs.annotate(xy=ohare,s="OHare",color='white')
axs.annotate(xy=loop,s="Loop",color='white')

# annotate dist: gpd.geoseries.GeoSeries(chi_shapefile[chi_shapefile['NAMELSAD']=='State House District 45'].geometry).centroid
axs.annotate(xy=(-88.10863773846053, 41.90002038299817), s="Dist 45", color='white')

# set axes extents from shapefile
yl=41.65;yu=42.3
xu=-87.47;xl=-88.3
axs.set_extent([xl,xu,yl,yu],crs= crs_new)

# title
axs.set_title('NO2 at 1 PM, Aug. 2018')

#add colorbar and label
cbar=plt.colorbar(cs,boundaries=np.arange(vmin,11))
#cbar.ax.set_ylabel('100 * ' +ncfile[0][var].units)
cbar.set_ticks(np.arange(vmin, vmax, 10))

# add state lines
import cartopy.feature as cfeature
states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',facecolor='none')

axs.add_feature(cfeature.STATES, edgecolor='black')

#add chi neighbs
#rep_districts_shapefile.plot(ax=axs, transform= crs_new,facecolor='None',edgecolor='grey',alpha=0.5)

#add epa monitors
#where are EPA monitors in CHI area
latttt=[41.920009, 42.062053, 41.755832, 41.855243,41.984332, 41.801180, 41.751400]
lonbbb=[-87.672995,-87.675254,-87.545350,-87.752470,-87.792002,-87.832349, -87.713488]
axs.scatter(lonbbb, latttt, marker = '*', color = 'white', s = 30)

plt.savefig(var+'_dist45.pdf')

plt.show()

