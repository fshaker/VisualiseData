import pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
pickle_out = open('TrafficData.pkl', "rb")
speedData = pickle.load(pickle_out)#this is an array of size (43,253,287) (number of days, number of links, samples per day)
pickle_out.close()
print(speedData.shape)
meanSpeed = np.mean(speedData, axis=0)
stdSpeed = np.std(speedData, axis=0)
# print((meanSpeed).astype(int))

# print((stdSpeed).astype(int))
binSpeed = speedData[1,:,:]>(meanSpeed-stdSpeed)
# print(binSpeed)
print(binSpeed.shape)

# plt.plot(binSpeed[:,0])
# plt.show()