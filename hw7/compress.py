import numpy as np
import pickle
import os
import torch
import sys

original = sys.argv[1]
print(f"\noriginal cost: {os.stat(original).st_size} bytes.")
params = torch.load(original)

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param
    #torch.save(custom_dict, fname)
    pickle.dump(custom_dict, open(fname, 'wb'))


encode8(params, '8_bit_model.pkl')
print(f"8-bit cost: {os.stat('8_bit_model.pkl').st_size} bytes.")