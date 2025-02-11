#!/bin/bash
if [ "$#" -ne 3 ]; then
  echo "Uso: $0 <crystal> <beam> <channel>"
  exit 1
fi

crystal=$1
beam=$2
channel=$3

for angle in 70 80 90 100 120 140 160 180; do
    python3 CherenkovYieldFit.py --crystal "$crystal" --beam "$beam" --angle "$angle" --channel "$channel" --Laser
done
