# Copied from http://superfluoussextant.com/making-gifs-with-python.html tytyty

import moviepy.editor as mpy

gif_name = 'NO2'
fps = 10
file_list = ['NO2_day%i_hour%i.png'%(j,k) for j in range(5) for k in range(24)]
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)
