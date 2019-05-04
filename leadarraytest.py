import numpy as np
from tempfile import TemporaryFile

# data = np.load("outfile.npy")
# patterns = data[0:100, 50:52, (0,1,2,5,6,7,8,9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28,29,30, 33,34,35,36,37, 40,41,42)]
# # ___________________
# print(patterns)
# print(data.shape)

a = np.arange(0,24)
a = a.reshape([2,3,4])

b=np.arange(5,15)

# outfile = TemporaryFile()
# np.savez("outfile", a=a, b=b)

r=np.load("outfile.npz")
