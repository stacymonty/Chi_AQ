# model validation table for CHEMICALS

import pandas as pd
import numpy as np
import scipy.stats as st
import wrf
from netCDF4 import Dataset
import glob,os
import matplotlib.pyplot as plt
import scipy.stats as st

#input
#dirToWRF_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/'
#dirToWRF_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/'

dirToWRF_d02='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/'
dirToWRF_d03='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/'
dir = '/projects/b1045/montgomery/'

fnames = ['NO2_d03_2018_8_EPA_CMAQ_Combine.csv','NO2_d03_2019_1_EPA_CMAQ_Combine.csv',
'NO2_d02_2018_8_EPA_CMAQ_Combine.csv','NO2_d02_2019_1_EPA_CMAQ_Combine.csv',
'O3_d03_2018_8_EPA_CMAQ_Combine.csv','O3_d03_2019_1_EPA_CMAQ_Combine.csv',
'O3_d02_2018_8_EPA_CMAQ_Combine.csv','O3_d02_2019_1_EPA_CMAQ_Combine.csv',
'PM25_TOT_d03_2018_8_EPA_CMAQ_Combine.csv','PM25_TOT_d03_2019_1_EPA_CMAQ_Combine.csv',
'PM25_TOT_d02_2018_8_EPA_CMAQ_Combine.csv','PM25_TOT_d02_2019_1_EPA_CMAQ_Combine.csv']

# functions
def stats(data,prediction):
	x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
	mu_d,mu_p = np.mean(x),np.mean(y)
	bias = np.sum(x-y)/len(x)
	rmse = np.sqrt(np.mean((y-x)**2))
	r,p = st.pearsonr(x,y)
	return mu_d,mu_p,bias,rmse,r,p

# functions
def stats_normalized(data,prediction):
	x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
	mu_d,mu_p = np.mean(x),np.mean(y)
	nmb = np.sum(y-x)/np.sum(x)*100
	nme = np.sum(np.abs(y-x))/np.sum(x)*100
	r,p = st.pearsonr(x,y)
	return mu_d,mu_p,nmb,nme,r,p

def pull_winds(dirwrf,fnames,xx,yy):
	fws,fwd = [],[]
	for q in range(len(fnames)):
		wrfout = wrf.g_uvmet.get_uvmet10_wspd_wdir(Dataset(dirwrf + fnames[q]),wrf.ALL_TIMES)
		winds = [[wrfout.data[0][hour][xx[i]][yy[i]] for i in range(len(xx))] for hour in range(24)] 
		winddir = [[wrfout.data[1][hour][xx[i]][yy[i]] for i in range(len(xx))] for hour in range(24)]
		fws.append(winds)
		fwd.append(winddir)
	# return
	return fws,fwd


# start
out = []
out2 = []
indnames = ['NO2_d03_Summer','NO2_d03_Winter','NO2_d02_Summer','NO2_d02_Winter','O3_d03_Summer','O3_d03_Winter','O3_d02_Summer','O3_d02_Winter','PM25_d03_Summer','PM25_d03_Winter','PM25_d02_Summer','PM25_d02_Winter',]

for i in range(len(fnames)):
	f = pd.read_csv(dir+'/CMAQcheck/'+fnames[i])
	if i>3 and i<8:  
		s = stats(f['Sample Measurement']*1000,f['CMAQ'])
		s2 = stats_normalized(f['Sample Measurement']*1000,f['CMAQ'])
	else: 
		s = stats(f['Sample Measurement'],f['CMAQ'])
		s2 = stats_normalized(f['Sample Measurement'],f['CMAQ'])
	out.append(s)
	out2.append(s2)
	#if len(f[f['level_0']=='2018-08-01 00:00:00']) >0: print(indnames[i]+'| number of stations = %i'%len(f[f['level_0']=='2018-08-01 00:00:00']))
	#if len(f[f['level_0']=='2019-01-02 00:00:00']) >0: print(indnames[i]+'winter| number of stations = %i'%len(f[f['level_0']=='2019-01-02 00:00:00']))
	print('%s| number of stations = %.1f'%(indnames[i],len(f['Longitude'].unique())-1))

out = pd.DataFrame(out)
out.columns=['mu_d','mu_p','bias','rmse','r','p']

out.index=['NO2_d03_Summer','NO2_d03_Winter','NO2_d02_Summer','NO2_d02_Winter','O3_d03_Summer','O3_d03_Winter','O3_d02_Summer','O3_d02_Winter','PM25_d03_Summer','PM25_d03_Winter','PM25_d02_Summer','PM25_d02_Winter',]

out

out2 = pd.DataFrame(out2)
out2.columns=['mu_d','mu_p','MB','NME','r','p']
out2.index=['NO2_d03_Summer','NO2_d03_Winter','NO2_d02_Summer','NO2_d02_Winter','O3_d03_Summer','O3_d03_Winter','O3_d02_Summer','O3_d02_Winter','PM25_d03_Summer','PM25_d03_Winter','PM25_d02_Summer','PM25_d02_Winter',]

out2.to_csv('~/chemicals_normalized.csv')



# model validation name for  meteorology
#getting wrf windspeed/directions: 
# NEED TO DO FOR WINTER
# knots to m/s - knots/1.9438444924406
sim = 'output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852'

windstn = pd.read_csv(dir+'/WRFcheck/'+sim+'/wrfcheck_withstations_'+sim+'_Wind.csv',index_col=0)
windDirstn = pd.read_csv(dir+'WRFcheck/'+sim+'/wrfcheck_withstations_'+sim+'_WindDir.csv',index_col=0)
times = pd.read_csv(dir+'WRFcheck/'+sim+'/completeddata_mini_extras2.csv')
#check winter and summer times for station index

xx_d02,yy_d02 = np.array(windstn['xx_d02']),np.array(windDirstn['yy_d02'])
xx_d03,yy_d03 = np.array(windstn[windstn['in_d03']==True]['xx_d03']),np.array(windstn[windstn['in_d03']==True]['yy_d03'])

# 10*24+1:-24-9
# :744

fws_stn_d03 = np.array(windstn[windstn['in_d03']==True].T[10*24+1:-24-9],dtype='float32')
fwd_stn_d03 = np.array(windDirstn[windDirstn['in_d03']==True].T[10*24+1:-24-9],dtype='float32')

#fws_stn_d02 = np.array(windstn[windstn['in_d02']==True].T[:744],dtype='float32')
#fwd_stn_d02 = np.array(windDirstn[windDirstn['in_d02']==True].T[:744],dtype='float32')
fws_stn_d02 = np.array(windstn[windstn['in_d02']==True].T[10*24+1:-24-9],dtype='float32')
fwd_stn_d02 = np.array(windDirstn[windDirstn['in_d02']==True].T[10*24+1:-24-9],dtype='float32')


filenames_d02=[]
os.chdir(dirToWRF_d02)
for file in glob.glob("wrfout_d01_*"):
    filenames_d02.append(file)

filenames_d02.sort()

# $1 Get WRF file names
filenames_d03=[]
os.chdir(dirToWRF_d03)
for file in glob.glob("wrfout_d01_*"):
    filenames_d03.append(file)

filenames_d03.sort()

# pull wind and dir
fws_d02,fwd_d02 = pull_winds(dirToWRF_d02,filenames_d02[10:-1],xx_d02,yy_d02)
fws_d03,fwd_d03 = pull_winds(dirToWRF_d03,filenames_d03[10:-1],xx_d03,yy_d03)


# make array and reshape
fws_d03= np.asarray(fws_d03)
fws_d03 = np.array([fws_d03[i][x] for i in range(len(filenames_d02[10:-1])) for x in range(24)])
fwd_d03 = np.array([fwd_d03[i][x] for i in range(len(filenames_d02[10:-1])) for x in range(24)])

fws_d02=np.asarray(fws_d02)
fws_d02 = np.array([fws_d02[i][x] for i in range(len(filenames_d02[10:-1])) for x in range(24)])
fwd_d02 = np.array([fwd_d02[i][x] for i in range(len(filenames_d02[10:-1])) for x in range(24)])

#d03

ld03= windstn[windstn['in_d03']==True]
ld03 = ld03[['lat','lon']]
ld03.reset_index(inplace=True,drop=True)

ld02= windstn[windstn['in_d02']==True]
ld02 = ld02[['lat','lon']]
ld02.reset_index(inplace=True,drop=True)



#write out files
if write ==True:
	pd.DataFrame(fws_d03/1.9438444924406).append(ld03.T).to_csv('~/wrf_winds_jan_d03.csv')
	pd.DataFrame(fws_d02/1.9438444924406).append(ld02.T).to_csv('~/wrf_winds_jan_d02.csv')
	pd.DataFrame(fwd_d03).append(ld03.T).to_csv('~/wrf_winddir_jan_d03.csv')
	pd.DataFrame(fwd_d02).append(ld02.T).to_csv('~/wrf_winddir_jan_d02.csv')


b=fws_d03.ravel()
a=fws_stn_d03.ravel()/1.9438444924406
stwspd_d03 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

b=fwd_d03.ravel()
a=fwd_stn_d03.ravel()
stwdir_d03 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

# d02

b=fws_d02.ravel()
a=fws_stn_d02.ravel()/1.9438444924406
stwspd_d02 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

b=fwd_d02.T.ravel()
a=fwd_stn_d02.ravel()
stwdir_d02 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

dfout=pd.DataFrame([stwspd_d02,stwdir_d02,stwspd_d03,stwdir_d03])
dfout.index = ['speed_d02','dir_d02','speed_d03','dir_d03']
dfout.columns = ['mu_d','mu_p','MB','MSE','r','p']


dfout.to_csv('~/windmetrics_summer_normalized.csv')

##-----------
#  get temperature and RH shit from combine aconc
# pull station data again
tmpstn = pd.read_csv('/home/asm0384/WRFcheck/'+sim+'/wrfcheck_withstations_'+sim+'_012019.csv',index_col=0)
rhstn = pd.read_csv('/home/asm0384/WRFcheck/'+sim+'/wrfcheck_withstations_'+sim+'_RH.csv',index_col=0)

xx_d02,yy_d02 = np.array(tmpstn['xx_d02']),np.array(tmpstn['yy_d02'])
xx_d03,yy_d03 = np.array(tmpstn[tmpstn['in_d03']==True]['xx_d03']),np.array(tmpstn[tmpstn['in_d03']==True]['yy_d03'])

# check completedatamini for the times associated with the indices
wint_ind = ':744'
sum_ind = '11*24+1:-9'

temp_stn_d03 = np.array(tmpstn[tmpstn['in_d03']==True].T[11*24+1:-9],dtype='float32')
rh_stn_d03 = np.array(rhstn[rhstn['in_d03']==True].T[11*24+1:-9],dtype='float32')

temp_stn_d02 = np.array(tmpstn[tmpstn['in_d02']==True].T[11*24+1:-9],dtype='float32')
rh_stn_d02 = np.array(rhstn[rhstn['in_d02']==True].T[11*24+1:-9],dtype='float32')

# pull aconc files
filenames_d02=[]
os.chdir(dirToWRF_d02+'/postprocess/')
for file in glob.glob("COMBINE_ACONC*"):
    filenames_d02.append(file)

filenames_d02.sort()

# $1 Get WRF file names
filenames_d03=[]
os.chdir(dirToWRF_d03+'/postprocess/')
for file in glob.glob("COMBINE_ACONC*"):
    filenames_d03.append(file)

filenames_d03.sort()


def get_temp_rh(dirToWRF_d02,filenames_d02,var,xx,yy):
	d2=[]
	for q in range(len(filenames_d02)):
		nc = Dataset(dirToWRF_d02 +'/postprocess/'+ filenames_d02[q])
		d=[[nc[var][hour][0][xx[i]][yy[i]] for i in range(len(xx))] for hour in range(24)]
		d2.append(d)
		#
	d2=np.asarray(d2)
	d2 = np.array([d2[i][x] for i in range(len(filenames_d02)) for x in range(24)])
	#
	return d2

temp_d02 = get_temp_rh(dirToWRF_d02,filenames_d02[0:],'SFC_TMP',xx_d02,yy_d02)
temp_d03 = get_temp_rh(dirToWRF_d03,filenames_d03[0:],"SFC_TMP",xx_d03,yy_d03)
rh_d02 = get_temp_rh(dirToWRF_d02,filenames_d02[0:],'RH',xx_d02,yy_d02)
rh_d03 = get_temp_rh(dirToWRF_d03,filenames_d03[0:],'RH',xx_d03,yy_d03)

#d03

ld03= tmpstn[tmpstn['in_d03']==True]
ld03 = ld03[['lat','lon']]
ld03.reset_index(inplace=True,drop=True)

ld02= rhstn[rhstn['in_d02']==True]
ld02 = ld02[['lat','lon']]
ld02.reset_index(inplace=True,drop=True)


#write out files
if write ==True:
	pd.DataFrame(temp_d03).append(ld03.T).to_csv('~/wrf_t2_jan_d03.csv')
	pd.DataFrame(temp_d02).append(ld02.T).to_csv('~/wrf_t2_jan_d02.csv')
	pd.DataFrame(rh_d03).append(ld03.T).to_csv('~/wrf_rh_jan_d03.csv')
	pd.DataFrame(rh_d02).append(ld02.T).to_csv('~/wrf_rh_jan_d02.csv')

# do stats

b=temp_d02.ravel()
a=(temp_stn_d02.ravel()-32)*5/9
st_temp_d02 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

b=temp_d03.ravel()
a=(temp_stn_d03.ravel()-32)*5/9
st_temp_d03 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])


b=rh_d02.ravel()
a=rh_stn_d02.ravel()
st_rh_d02 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])

b=rh_d03.ravel()
a=rh_stn_d03.ravel()
st_rh_d03 = stats_normalized(a[~np.isnan(b)],b[~np.isnan(b)])


dfout=pd.DataFrame([st_temp_d02,st_rh_d02,st_temp_d03,st_rh_d03])
dfout.index = ['temp_d02','rh_d02','temp_d03','rh_d03']
dfout.columns = ['mu_d','mu_p','MB','MSE','r','p']


dfout.to_csv('~/temp_rh_summermetrics_normalized.csv')


# now group by each station and get correlation

fnames = ['NO2_d03_2018_8_EPA_CMAQ_Combine.csv','NO2_d03_2019_1_EPA_CMAQ_Combine.csv',
'O3_d03_2018_8_EPA_CMAQ_Combine.csv','O3_d03_2019_1_EPA_CMAQ_Combine.csv',
'PM25_TOT_d03_2018_8_EPA_CMAQ_Combine.csv','PM25_TOT_d03_2019_1_EPA_CMAQ_Combine.csv']

level_2 = ['NO2_Sum','NO2_Wint','O3_Sum','O3_Wint','PM25_Sum','PM25_Wint']

for i in range(len(fnames)):
	fname = fnames[i]
	f = pd.read_csv(dir+'/CMAQcheck/'+fname)
	fl = f.groupby(['Latitude','Longitude'])[['CMAQ','Sample Measurement']].corr().iloc[0::2,-1]
	fl = fl.reset_index()
	fl['level_2']=level_2[i]
	print(fl)
	if i==0: final = fl
	else: final = final.append(fl)
# using 

final.to_csv('~/Station_correlations_EPA.csv')




