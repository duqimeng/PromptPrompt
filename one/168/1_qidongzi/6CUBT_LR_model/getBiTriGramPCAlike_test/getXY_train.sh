#!/bin/sh
rm result/*
rm statistic/*
rm statistic_2/*
rm classify_data/*
echo "now start"
python 1getLabelCharUniBiTri2cnt.py --file $1
python 2getClassify_data.py  --file $1
python 3getTF.py
python 3.5getTF2.py
python 4getFeatureLimit.py
python 5getXY.py
echo "finished"

