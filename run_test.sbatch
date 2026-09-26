#!/bin/bash
#SBATCH --job-name=sc-macro
#SBATCH --output=logs/sc-macro-%j.out
#SBATCH --error=logs/sc-macro-%j.err
#SBATCH --time=06:00:00            # full suite, generous budget
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1              # one GPU (only cupy targets it; jax/torch stay CPU)

set -eo pipefail

mkdir -p logs bench/out
cd "${SLURM_SUBMIT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Fail fast if the environment can't even import/parse the suite,
# rather than burning the time budget on a startup error.
python -m bench.run_macro --quick --backend numpy --seeds 0 \
    --out bench/out/macro_smoke.json --quiet

# Base operations (space ops + operator apply), sizes up to 5000.
python -m bench.run_micro \
    --backend numpy --backend jax --backend torch --backend cupy \
    --max-size 5000 \
    --out bench/out/micro_full.json

# Macrobenchmark suite, restricted to problem sizes up to ~5000.
# ('all' would also pull pdhg (50000,20000) and power_lanczos d=10000/100000.)
SIZES="n=64,n=128,n=256,b16_d4,b64_d8,b256_d16,n=32,(2000,1000),d=256,d=1024,d=4096_depth8,(5000,2000),s4_d4,s4_d8,s8_d16"

python -m bench.run_macro \
    --backend numpy --backend jax --backend torch --backend cupy \
    --check-level both \
    --seeds 0,1,2,3 \
    --sizes "${SIZES}" \
    --out bench/out/macro_full.json

# Render one HTML dashboard from both JSON artifacts (micro + macro).
python -m bench.dashboard \
    --input bench/out/micro_full.json \
    --input bench/out/macro_full.json \
    --out bench/out/dashboard.html
