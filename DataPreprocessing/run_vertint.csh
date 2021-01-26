!/bin/csh

#For creating VCD from CMAQ output -- needs mcip files and cmaq cctm files

# For future runs: check indir, fname_start, begdate + enddate
# Make sure indir/column and indir/mcip exists
#
# Location of top directory CONC and MCIP files
setenv indir /projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852
# # Location to create vert files
setenv outdir $indir/column
# # Location of CONC files
setenv concdir $indir
# # Location of METCRO3D files
setenv mcipdir $indir/mcip
# How the netcdffile is named at the front
setenv fname_start CCTM_CONC_v3852_

#
# #Starting and ending times (inclusive) of times
setenv begdate_g "2019-01-01"   # YYYYMMDD
setenv enddate_g "2019-02-01"   # YYYYMMDD
#
# # beg date julian
setenv begdate_j  `date -ud "${begdate_g}" +%Y%j`
# # end date julian
setenv enddate_j  `date -ud "${enddate_g}" +%Y%j`
# # curr date (updated in loop) julian
setenv curdate_j  $begdate_j
# # curr date (updated in loop) gregorian
setenv curdate_g  $begdate_g
# # curr date (updated in loop) gregorian
setenv curdate_g_f `date -ud "${curdate_g}" +%Y%m%d`
#
# # Main loop
while ( $curdate_j <= $enddate_j)
#
# # Set name of input file
setenv infile $indir/$fname_start$curdate_g_f".nc"
# # Name of output file
setenv outfile $outdir/$fname_start$curdate_g_f"_column.nc"
# # Name of Metfile
setenv metfile $mcipdir/"METCRO3D_Chicago_LADCO_"$curdate_g".nc"

vertintegral<< TEST_DONE
infile



metfile
outfile
TEST_DONE


setenv curdate_g  `date -ud "${curdate_g}+1days" +%Y-%m-%d`
setenv curdate_j  `date -ud "${curdate_g}" +%Y%j`
setenv curdate_g_f `date -ud "${curdate_g}" +%Y%m%d`

#TEST_DONE

echo "----------------------------- "
echo $curdate_g

end
