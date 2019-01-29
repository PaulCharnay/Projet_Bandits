# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:35:12 2018

@author: Paul Charnay
"""

import numpy as np

records_array = np.array([1, 4, 3, 2, 2])
vals, inverse, count = np.unique(records_array, return_inverse=True, return_counts=True)

idx_vals_repeated = np.where(count > 1)[0]
vals_repeated = vals[idx_vals_repeated]

rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
_, inverse_rows = np.unique(rows, return_index=True)
res = np.split(cols, inverse_rows[1:])

print(res)



