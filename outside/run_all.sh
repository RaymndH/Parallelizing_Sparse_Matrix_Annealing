#!/bin/bash
# outside solvers on c=18 planted graphs; each run has a wall-clock limit. usage: run_all.sh
cd /data/Raymond/Parallelizing_Sparse_Matrix_Annealing/tools; source venv_tools/bin/activate
O=../outside; K=src_kissat/build/kissat
for n in 1e4 1e5; do
  case $n in 1e4) G=../SimAnn/graph/N-1e4_c-18.0_q-5; L=900;; 1e5) G=../gen1e5/graph/N-1e5_c-18.0_q-5; L=1800;; esac
  echo "== kissat $n (limit ${L}s)"; ( time timeout $L taskset -c 100-100 $K -q g$n.cnf | head -1 ) 2>&1 | grep -E "^s |real"
  echo "== cpsat $n (limit ${L}s, 8 workers)"; taskset -c 101-108 python run_cpsat.py $G $L 8
  echo "== neal $n (1000 sweeps)"; taskset -c 109 python run_neal.py $G 1000 1
done
