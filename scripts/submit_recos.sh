for dir in {15..20}
do
for file in {0..49}
do
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A cyberlamp \
#-l qos=cl_open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A dfc13_b_g_sc_default \
#-l qos=cl_open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A dfc13_a_g_sc_default \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A open \
    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A cyberlamp \
-l nodes=1:ppn=1 \
-l pmem=8000mb \
-l walltime=24:00:00 \
-N r$dir.$file \
-o /gpfs/scratch/pde3/retro/log/$dir.$file.log \
-e /gpfs/scratch/pde3/retro/log/$dir.$file.err
done
done
