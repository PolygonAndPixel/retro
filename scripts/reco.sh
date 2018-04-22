#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

events_base="$1"
start_idx="$2"
outdir="$3"

mkdir -p "$outdir"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""

#proto="/gpfs/scratch/jll1062/retro_tables/stacked"
#tmpl_lib="--template-library /gpfs/scratch/jll1062/retro_tables/ckv_dir_templates.npy"

#proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"

# ET:
#proto="/data/icecube/retro_tables/large_5d_notilt_combined/stacked"
#tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"
# ACI:
proto="/gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/stacked/"
tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"

#no_noise="--no-noise"
no_noise=""

importance_sampling="--importance-sampling"
#importance_sampling=""

consteff="--const-eff"
#consteff=""

#cascade_kernel="point_ckv"
#cascade_samples=""
cascade_kernel="one_dim"
cascade_samples="--cascade-samples 500"


#kernprof -l -v ~/src/retro/retro/reco.py \
#gdb -ex r --args python ~/retro/retro/reco.py \
python ~/retro/retro/reco.py \
    --outdir "$outdir" \
    --spatial-prior SPEFit2 \
    --temporal-prior SPEFit2 \
    --cscd-energy-prior log_uniform \
    --cscd-energy-lims 0.1,1000  \
    --trck-energy-prior uniform \
    --trck-energy-lims 0.1,500  \
    \
    $importance_sampling \
    --max-modes 1 \
    $consteff \
    --n-live 1000 \
    --evidence-tol 0.1 \
    --sampling-eff 0.5 \
    --max-iter 10000 \
    --seed 0 \
    \
    --dom-tables-kind "ckv_templ_compr" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    --gcd "GeoCalibDetectorStatus_IC86.55697_corrected_V2.pkl" \
    --norm-version "binvol2" \
    $tmpl_lib \
    --step-length 1.0 \
    $no_noise \
    \
    --cascade-kernel $cascade_kernel \
    --track-kernel "const_e_loss" \
    --track-time-step 1.0 \
    \
    --events-base "$events_base" \
    --start-idx "$start_idx" \
    --num-events 100 \
    --truth \
    --pulses "OfflinePulses" \
    --recos "SPEFit2" \
    --triggers "I3TriggerHierarchy" \
    --hits "pulses/OfflinePulses" \
    --angsens-model "h2-50cm"

wait
