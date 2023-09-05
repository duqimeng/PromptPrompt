#!/bin/sh
echo "now start"
rm classify_data/*
python 2getClassify_data.py  --file $1 --which "test"
python 5getXY.py  --which "test" --LiYu "True"
echo "finished"

