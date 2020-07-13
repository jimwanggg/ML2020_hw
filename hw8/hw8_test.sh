#!/bin/bash
wget https://github.com/jimwanggg/StoreModel/releases/download/seq2seq/model.ckpt
python test.py $1 $2