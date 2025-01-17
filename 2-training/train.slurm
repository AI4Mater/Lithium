#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100G

source /etc/profile
ulimit -s unlimited
ulimit -l unlimited
module purge

conda deactivate
conda activate mace

export MKL_THREADING_LAYER=INTEL
echo "Job started on `hostname` at `date`"
nvidia-cuda-mps-control -d #start NVIDIA MPS daemon service

mpirun -n 1 --map-by socket:PE=1 --bind-to core mace_run_train \
    --name="Li_MACE" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=4.5 \
    --max_L=2 \
    --batch_size=1 \
    --max_num_epochs=200 \
    --patience=5 \
    --swa \
    --ema \
    --restart_latest \
    --device=cuda \
    --default_dtype=float64 \
    --error_table='PerAtomMAE' \
    --seed=1214 \
    --save_cpu

echo "quit" | nvidia-cuda-mps-control #quit MPS daemon
echo "Job ended on `hostname` at `date`"