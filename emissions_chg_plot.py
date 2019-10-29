#!/usr/bin/env python

------------------------------------------
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

# files
dir='/projects/b1045/jschnell/ForStacy/'
ll='latlon_ChicagoLADCO_d03.nc'
emis='emis_20180801_noSchoolnoBusnoRefuse_minus_base.nc'
emis='emis_20180801_noSchool_minus_base.nc'
ll=Dataset(dir+ll,'r')
lat,lon=ll['lat'][:],ll['lon'][:]

path='/home/asm0384/shapefiles/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'

# Start pulling and cropping data
chi  = gpd.GeoDataFrame.from_file(path)


#pull in files and variables
ncfile= Dataset(dir+emis,'r')
df_lat,df_lon=pd.DataFrame(lat),pd.DataFrame(lon)
no2= pd.DataFrame(Dataset(dir+emis,'r')['NO2'][13][0][:])*10e2
df=pd.DataFrame(no2[:])
no2_drop=df.loc[~(df==0).all(axis=1)]

# drop outside parts that are 0 in the array
data= np.array(df.loc[~(df==0).all(axis=1)])
lat= np.array(df_lat.loc[~(df==0).all(axis=1)])
lon= np.array(df_lon.loc[~(df==0).all(axis=1)])


# files

emis1='emis_20180801_noSchoolnoBusnoRefuse_minus_base.nc'

#pull in files and variables
ncfile1= Dataset(dir+emis1,'r')

no21= pd.DataFrame(Dataset(dir+emis1,'r')['NO2'][13][0][:])*10e2
df1=pd.DataFrame(no21[:])

# drop outside parts that are 0 in the array
data1= np.array(df1.loc[~(df1==0).all(axis=1)])
lat1= np.array(df_lat.loc[~(df1==0).all(axis=1)])
lon1= np.array(df_lon.loc[~(df1==0).all(axis=1)])
data=data-data1

crs_new = ccrs. AlbersEqualArea(central_longitude=(chi.bounds.mean().minx+chi.bounds.mean().maxx)/2)


# get shape outside
union=gpd.GeoSeries(unary_union(chi.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])

# make fig object
fig, axs = plt.subplots(subplot_kw={'projection': crs_new},figsize=(6, 6))

#set up data for plotting via levels
vmax=pd.DataFrame(data).max().max()
vmin= int(pd.DataFrame(data).min().min())
vmax=-.5
vmin=-1.5
levels = np.linspace(vmin, int(vmax), 15)


# get rid of values outside the levels we are contouring to
data[pd.DataFrame(data)<vmin]=vmin


# set boundary as outer extent by making a matplotlib path object and adding that geometry
# i think setting the boundary before you plot the data actually crops the data to the shape, so set ax first
axs.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
axs.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')

#plot the gridded data by using contourf
cs=plt.contourf(lon,lat,data,cmap= "inferno", transform=crs_new, levels=levels)

# add landmarks with scatterplot
midway= 41.7868, -87.7522
ohare = 41.9742, -87.9073
loop = 41.8786, -87.6251
#plt.scatter(pd.DataFrame([midway,ohare,loop])[1],pd.DataFrame([midway,ohare,loop])[0],marker = '*',color='white')

# set axes extents from shapefile
x=[min(chi.bounds.minx), max(chi.bounds.maxx)] 
y=[min(chi.bounds.miny), max(chi.bounds.maxy)]
axs.set_extent([x[0]-.03,x[1]+.03,y[0]-.03,y[1]+.03],crs= crs_new)
axs.set_title('1 PM Change in Emissions from Scenario 2')


#add colorbar and label
cbar=plt.colorbar(cs,boundaries=np.arange(vmin,11))
cbar.ax.set_ylabel('moles/sec')
cbar.set_ticks(np.arange(vmin, int(vmax),.5))



#EPA Sensor Scatter


# get rid of values outside the levels we are contouring to
fig, axs = plt.subplots()

chi.plot(ax=axs,color='lightgrey',linewidth=100)

# set boundary as outer extent by making a matplotlib path object and adding that geometry
# i think setting the boundary before you plot the data actually crops the data to the shape, so set ax first

labels=['NO2','Ozone','NO2','Ozone','Ozone','PM10','NO2']
values=[1,2,1,2,2,3,1]

latttt=[41.920009, 42.062053, 41.755832, 41.855243,41.984332, 41.801180, 41.751400]

lonbbb=[-87.672995,-87.675254,-87.545350,-87.752470,-87.792002,-87.832349, -87.713488]

axs.scatter(lonbbb, latttt,c=values,s=100,label=labels)

fig.patch.set_visible(False)
axs.axis('off')

#legend1 = axs.legend(*scatter.legend_elements(num=4),
#                    loc="outer left", title="Ranking")

#axs.add_artist(legend1)

#axs.legend()

plt.show()


