# python 3.7.4 @hehaoran
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


# Tensor Decomposition
x = tl.tensor(np.arange(20).reshape(3,4,2).astype(float))
core,factors = parafac(x,rank = 3)
print(factors)

