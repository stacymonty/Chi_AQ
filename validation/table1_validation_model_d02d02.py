# model validation table for CHEMICALS

import pandas as pd
import numpy as np
import scipy.stats as st


def stats(data,prediction):
	x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
	mu_d,mu_p = np.mean(x),np.mean(y)
	bias = np.sum(x-y)/len(x)
	rmse = np.sqrt(np.mean((y-x)**2))
	r,p = st.pearsonr(x,y)
	return mu_d,mu_p,bias,rmse,r,p



fnames = ['NO2_d03_2018_8_EPA_CMAQ_Combine.csv','NO2_d03_2019_1_EPA_CMAQ_Combine.csv',
'NO2_d02_2018_8_EPA_CMAQ_Combine.csv','NO2_d02_2019_1_EPA_CMAQ_Combine.csv',
'O3_d03_2018_8_EPA_CMAQ_Combine.csv','O3_d03_2019_1_EPA_CMAQ_Combine.csv',
'O3_d02_2018_8_EPA_CMAQ_Combine.csv','O3_d02_2019_1_EPA_CMAQ_Combine.csv',
'PM25_TOT_d03_2018_8_EPA_CMAQ_Combine.csv','PM25_TOT_d03_2019_1_EPA_CMAQ_Combine.csv',
'PM25_TOT_d02_2018_8_EPA_CMAQ_Combine.csv','PM25_TOT_d02_2019_1_EPA_CMAQ_Combine.csv']

out = []

for i in range(len(fnames)):
	f = pd.read_csv(fnames[i])
	if i>3 and i<8:  
		s = stats(f['Sample Measurement']*1000,f['CMAQ'])
	else: 
		s = stats(f['Sample Measurement'],f['CMAQ'])
	out.append(s)

out = pd.DataFrame(out)
out.columns=['mu_d','mu_p','bias','rmse','r','p']

out.index=['NO2_d03_Summer','NO2_d03_Winter','NO2_d02_Summer','NO2_d02_Winter','O3_d03_Summer','O3_d03_Winter','O3_d02_Summer','O3_d02_Winter','PM25_d03_Summer','PM25_d03_Winter','PM25_d02_Summer','PM25_d02_Winter',]

out