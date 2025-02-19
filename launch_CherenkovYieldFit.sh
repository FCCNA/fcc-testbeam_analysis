#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Uso: $0 <crystal> <beam>"
  exit 1
fi

crystal=$1
beam=$2

for angle in 0 20 40 50 60 70 80 90 100 110 120 130 140 160 180; do
  echo "${angle}"
  python3 CherenkovYieldFit.py --crystal "$crystal" --beam "$beam" --angle "$angle" --Laser
done