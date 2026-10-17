#!/usr/bin/env bash
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}   # repository root (default paths below are relative to it)
# Way-number sweep x false-negative oracle for ProtoCLR. Resumable: run it again
# after an interruption and unfinished runs continue from resume.pt.
#   DATA=data/mini84 WAYS="5 10 20 50" SEEDS="1" PAR=4 WORKERS=5 bash run_protoclr.sh
set -uo pipefail
DATA=${DATA:-$REPO/data/mini84}
OUT=${OUT:-runs_pc}
WAYS=${WAYS:-"5 10 20 50"}
SEEDS=${SEEDS:-"1"}
PAR=${PAR:-4}
WORKERS=${WORKERS:-5}
ITERS=${ITERS:-150000}
EXTRA=${EXTRA:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

jobs=()
for s in $SEEDS; do for n in $WAYS; do
  jobs+=("$n 0 base $s" "$n 1 oracle $s")
done; done

run_one() {
  read -r n o name s <<<"$1"
  d="$OUT/N${n}_${name}_s${s}"
  if [ -f "$d/last.pt" ]; then echo "skip $d (done)"; return; fi
  mkdir -p "$d"; echo "start $d $(date +%H:%M)"
  OMP_NUM_THREADS=4 python "$HERE/protoclr_oracle.py" --data_root "$DATA" --out "$d" --n_way "$n" \
      --oracle_neg "$o" --seed "$s" --iters "$ITERS" --workers "$WORKERS" $EXTRA >> "$d/train.out" 2>&1
  echo "done  $d $(date +%H:%M)"
}
export -f run_one
export OUT DATA WORKERS ITERS EXTRA HERE
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}

python "$HERE/eval_protoclr.py" --data_root "$DATA" --runs "$OUT"/N* --ckpt best_val.pt
python "$HERE/summarize_protoclr.py" --runs_dir "$OUT" --ckpt best_val
