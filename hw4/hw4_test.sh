#!/bin/bash
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/bi1.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/first_strong.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/fourth_strong.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/second_strong.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/third_strong.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/w2v_with_nolabel.model
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/w2v_with_nolabel.model.trainables.syn1neg.npy
wget https://github.com/jimwanggg/StoreModel/releases/download/RNN/w2v_with_nolabel.model.wv.vectors.npy
python out.py $1 $2