#!/bin/bash
mkdir checkpoints
cd checkpoints
wget https://github.com/jimwanggg/StoreModel/releases/download/unsupervised/best.pth
wget https://github.com/jimwanggg/StoreModel/releases/download/unsupervised/baseline.pth
wget https://github.com/jimwanggg/StoreModel/releases/download/unsupervised/improved.pth
cd ..
python test.py $1 $2 $3