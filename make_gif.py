#libraires
import moviepy.editor as mpy
import os
import glob


filestartswith='NO2'
dir='~/'

# pull grid stuff
#Get number of files in directory
onlyfiles = next(os.walk(dir))[2]
onlyfiles = [x for x in onlyfiles if x.startswith(filestartswith)]
onlyfiles=sorted(onlyfiles)# so th

fstart=[filestartswith+dates[i] for i in range(len(dates))]
fnames=[]

for f in onlyfiles: 
	for fs in fstart: 
		if f.startswith(fs): fnames.append(f)# Copied from http://superfluoussextant.com/making-gifs-with-python.html tytyty


#
gif_name = 'NO2'
fps = 10
file_list = ['NO2_day%i_hour%i.png'%(j,k) for j in range(5) for k in range(24)]
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)
