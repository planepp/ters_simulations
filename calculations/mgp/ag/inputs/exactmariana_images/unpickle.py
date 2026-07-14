import pickle
import numpy as np

with open("hessian.pickle", "rb") as f:
    arr = pickle.load(f)
   
print(np.shape(arr))
np.savetxt("hessian.dat", arr)
