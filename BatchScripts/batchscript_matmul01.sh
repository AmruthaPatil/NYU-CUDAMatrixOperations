#!/bin/bash
#SBATCH --job-name=l4_q4
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:rtx8000:1

cd path/to/PartA
make clean
make 2>/dev/null
make
./matmult01 8
./matmult01 16
./matmult01 32

