import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error


def corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    x,y = x[~nas], y[~nas]
    corr = pearsonr(x,y)[0]
    bias = (np.array(y)-np.array(x)).mean()
    #mse = mean_squared_error(x,y)
    return corr,bias#,mse


def get_corrs(fnames):
   corrs=[]; bias=[]; mses=[]
#
   for i in range(len(fnames)):
       df=pd.read_csv(fnames[i])
       latlon=[str(df.Latitude[i]) + " " + str(df.Longitude[i]) for i in range(len(df))]
       df['latlon']=latlon
       if df['Units of Measure'].unique()[0]=='Parts per million': df['Sample Measurement']=df['Sample Measurement']*1000
       elif df['Units of Measure'].unique()[1]=='Parts per million': df['Sample Measurement']=df['Sample Measurement']*1000
       x,y = np.array(df['Sample Measurement']),np.array(df['CMAQ'])
       cor,bia=corr(x,y)
       corrs.append(cor); bias.append(bia); #mses.append(mse)
       
#
   return corrs, bias, mses



fnamesd02=['CO_d02_2018_8_EPA_CMAQ_Combine.csv',
'CO_d02_2019_1_EPA_CMAQ_Combine.csv',
'NO2_d02_2018_8_EPA_CMAQ_Combine.csv',
'NO2_d02_2019_1_EPA_CMAQ_Combine.csv',
'O3_d02_2018_8_EPA_CMAQ_Combine.csv',
'O3_d02_2019_1_EPA_CMAQ_Combine.csv',
'SO2_d02_2018_8_EPA_CMAQ_Combine.csv',
'SO2_d02_2019_1_EPA_CMAQ_Combine.csv']  

fnamesd03=['CO_d03_2018_8_EPA_CMAQ_Combine.csv',
'CO_d03_2019_1_EPA_CMAQ_Combine.csv',
'NO2_d03_2018_8_EPA_CMAQ_Combine.csv',
'NO2_d03_2019_1_EPA_CMAQ_Combine.csv',
'O3_d03_2018_8_EPA_CMAQ_Combine.csv',
'O3_d03_2019_1_EPA_CMAQ_Combine.csv',
'SO2_d03_2018_8_EPA_CMAQ_Combine.csv',
'SO2_d03_2019_1_EPA_CMAQ_Combine.csv']      


corrd02,biasd02,msed02= get_corrs(fnamesd02)
corrd03,biasd03,msed03= get_corrs(fnamesd03)

namesd03 =[fnamesd03[i].split('_EPA_CMAQ_Combine.csv')[0] for i in range(len(fnamesd03))]
namesd02=[fnamesd02[i].split('_EPA_CMAQ_Combine.csv')[0] for i in range(len(fnamesd02))]

chems=['Aug CO','Jan CO','Aug NO2','Jan NO2','Aug O3','Jan O3','Aug SO2','Jan SO2']

final=pd.DataFrame([chems,corrd02,biasd02,corrd03,biasd03]).T
final.columns=['chem/date','r2 d02','bias d02','r2 d03','bias d03']

pd.options.display.float_format = '{:,.2f}'.format
final



def get_corrs_monthly(fnames):
   corrs=[]; bias=[]; mses=[]
#
   for i in range(len(fnames)):
       df=pd.read_csv(fnames[i])
#for i in range(1):
       if df['Units of Measure'][0]=='Parts per million': df['Sample Measurement']=df['Sample Measurement']*1000
       df['date']=pd.to_datetime(df['level_0'])
       df = df.set_index('date').sort_index()
       df=df.groupby(['Longitude','Latitude']).mean()
       x,y = np.array(df['Sample Measurement']),np.array(df['CMAQ'])
       cor,bia,mse=corr(x,y)
       corrs.append(cor); bias.append(bia); mses.append(mse)
#
   return corrs, bias, mses


# 7day weekly means 
df=pd.read_csv(fnamesd03[0]); df['date']=pd.to_datetime(df['level_0']); df = df.set_index('date').sort_index(); df=df.groupby(['Longitude','Latitude']).resample('7d').mean()
x,y = np.array(df['Sample Measurement']),np.array(df['CMAQ'])
cor,bia,mse=corr(x,y)
df.plot.scatter('Sample Measurement','CMAQ',c='Longitude',colormap='viridis')

#7day rolling means
df=pd.read_csv(fnamesd03[0]); df['date']=pd.to_datetime(df['level_0']); df = df.set_index('date').sort_index(); df=df.groupby(['Longitude','Latitude']).rolling('3h').mean()
x,y = np.array(df['Sample Measurement']),np.array(df['CMAQ'])
cor,bia,mse=corr(x,y)
df.plot.scatter('Sample Measurement','CMAQ',c='Longitude',colormap='viridis')


# REDO THIS: Two sentences in the methods
# Subset d03 and d02 --> pull same stations, see how the r2 changes
# make daily and monthly
# here's a more holistic picture

# >> make january o3 gif
# Bias of 30 -- is there even any measurements for o3 in january
# i think the offset is just stepwise changed

# making the gifs
# throw area average hours together and rank
#--> almost the worst 95%ile, max

# d02-> d03 gifs do roads show up can you see individiaul power plants


# Only d03 stations with d02
#------------------------------------------------------

head=['State Code', 'County Code', 'Site Num', 'Parameter Code', 'POC', 'Latitude', 'Longitude', 'Datum', 'Parameter Name', 'Date Local', 'Time Local', 'Date GMT', 'Time GMT', 'Sample Measurement', 'Units of Measure', 'MDL', 'Uncertainty', 'Qualifier', 'Method Type', 'Method Code', 'Method Name', 'State Name', 'County Name', 'Date of Last Change','date']

def only_d03_corr(df,df2):
#for i in range(1):
    df['date']=pd.to_datetime(df['level_0'])
    df2['date']=pd.to_datetime(df2['level_0'])
    latlon=[str(df.Latitude[i]) + " " + str(df.Longitude[i]) for i in range(len(df))]
    latlon2=[str(df2.Latitude[i]) + " " + str(df2.Longitude[i]) for i in range(len(df2))]
    df['latlon']=latlon;
    df2['latlon']=latlon2;
    if df['Units of Measure'].unique()[0]=='Parts per million': df['Sample Measurement']=df['Sample Measurement']*1000; df2['Sample Measurement']=df2['Sample Measurement']*1000
    elif df['Units of Measure'].unique()[1]=='Parts per million': df['Sample Measurement']=df['Sample Measurement']*1000; df2['Sample Measurement']=df2['Sample Measurement']*1000
    m=pd.merge(df2,df,on=['latlon','date'],suffixes=('_d02', '_d03'))
    x,y,z = np.array(m['Sample Measurement_d02']),np.array(m['CMAQ_d02']),np.array(m['CMAQ_d03'])
    corrd02=corr(x,y)[:2]
    corrd03=corr(x,z)[:2]
    nstations=len(m.latlon.unique())
    return corrd02,corrd03,nstations,np.nanmean(x),y.mean(),z.mean()


corrd02,corrd03=[],[]
biasd02,biasd03=[],[]
nstation=[]
avgstn=[];avgcmq2=[];avgcmq3=[]
for i in range(len(fnamesd02)):
    df,df2=pd.read_csv(fnamesd03[i]),pd.read_csv(fnamesd02[i])
    c2,c3,ns,xm,ym,zm = only_d03_corr(df,df2)
    corrd02.append(c2[0]);corrd03.append(c3[0])
    biasd02.append(c2[1]);biasd03.append(c3[1]);
    nstation.append(ns)
    avgstn.append(xm); avgcmq2.append(ym); avgcmq3.append(zm)
    


chems=['Aug CO','Jan CO','Aug NO2','Jan NO2','Aug O3','Jan O3','Aug SO2','Jan SO2']

final=pd.DataFrame([chems,corrd02,biasd02,corrd03,biasd03,avgstn,avgcmq2,avgcmq3,nstation]).T

final.columns=['chem/date','r2 d02','bias d02','r2 d03','bias d03','avg stn','avg d02','avg d03','n station']

pd.options.display.float_format = '{:,.2f}'.format
final

