#!/bin/bash
#SBATCH --job-name=l4_q1
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:rtx8000:1

cd path/to/PartA
make clean
make 2>/dev/null
make
./vecadd00 500
./vecadd00 1000
./vecadd00 2000