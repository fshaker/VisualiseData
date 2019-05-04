import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(threshold=np.NaN)



weightsChange = np.load("E:\Fujitsu\Codes\BoltzmanMachine/WeightsChange_SW_DA_0.npy")

#print(weightsChange.shape)
# print(np.mean(weightsChange))

weights = np.zeros((1001,weightsChange.shape[1]), int)
weightsChangeCompl = np.zeros((1000, weightsChange.shape[1]), int)
weights[0,:]=2*weightsChange[0,:]
for i in range(100):
    #print(i)
    weightsChange = np.load("E:\Fujitsu\Codes\BoltzmanMachine/WeightsChange_SW_DA_"+str(i)+".npy")
    for j in range(weightsChange.shape[0]):
        weights[i*10+j+1,:] = weights[i*10+j,:] + 2*weightsChange[j,:]
        weightsChangeCompl[i*10+j,:] = weightsChange[j,:]



print(weights)


plt.figure()
ax = plt.gca()
im = ax.imshow(weightsChangeCompl[0:100,:])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size='5%', pad=0.05)
plt.colorbar(im, cax = cax, ticks = [-1, 0, 1])
# plt.title("Weights Change in each iteration for first 100 connections")
# plt.xlabel("Connection number")
# plt.ylabel("Iteration")
plt.savefig('WeightsChange_SW_DA_0.jpg', bbox_inches='tight')
plt.close()

for i in range(29):
    plt.figure()
    plt.plot(weights[0:400,i])
    plt.savefig('SoftDAsingleConnectionWeightChange_'+str(i))
    plt.close()
######################################################################################
weightsChange = np.load("E:\Fujitsu\Codes\BoltzmanMachine/BMWeightsChange_0.npy")

#print(weightsChange.shape)
# print(np.mean(weightsChange))

weights = np.zeros((1001,weightsChange.shape[1]), int)
weightsChangeCompl = np.zeros((1000, weightsChange.shape[1]), int)
weights[0,:]=2*weightsChange[0,:]
for i in range(100):
    #print(i)
    weightsChange = np.load("E:\Fujitsu\Codes\BoltzmanMachine/BMWeightsChange_"+str(i)+".npy")
    for j in range(weightsChange.shape[0]):
        weights[i * 10 + j + 1, :] = weights[i * 10 + j, :] + 2*weightsChange[j,:]
        weightsChangeCompl[i*10+j,:] = weightsChange[j,:]


# change the order of the connections to match the Fujitsu one.
ind = np.asarray([29, 19, 23, 26, 28, 5, 9, 12, 14, 18, 22, 25, 27, 4, 8, 11, 13, 17, 21, 24, 16, 20, 15, 3, 7, 10, 2, 6, 1])-1
weightsChangeCompl = weightsChangeCompl[:,ind]
print(weights)


plt.figure()
ax = plt.gca()
im = ax.imshow(weightsChangeCompl[0:100,:])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size='5%', pad=0.05)
plt.colorbar(im, cax = cax, ticks = [-1, 0, 1])
# plt.title("Weights Change in each iteration for first 100 connections")
# plt.xlabel("Connection number")
# plt.ylabel("Iteration")
plt.savefig('BMWeightsChange_0.jpg', bbox_inches='tight')
plt.close()

for i in range(29):
    plt.figure()
    plt.plot(weights[0:400,i])
    plt.savefig('singleConnectionWeightChange_'+str(i))
    plt.close()
# weightsChange = np.load("E:/Fujitsu/AnomallyDetectionProject/weightsChange_small_0.npy")
# print(weightsChange.shape)
# # print(np.mean(weightsChange))
# weights = np.zeros((1,weightsChange.shape[1]), int)
# weightsChangeCompl = np.zeros((1370, weightsChange.shape[1]), int)
# for i in range(137):
#     weightsChange = np.load("E:/Fujitsu/AnomallyDetectionProject/weightsChange_small_"+str(i)+".npy")
#     for j in range(weightsChange.shape[0]):
#         weights = weights + 2*weightsChange[j,:]
#         weightsChangeCompl[i*10+j,:] = weightsChange[j,:]
#     #if(i==9):
#     w=np.zeros((10,10), int)
#     for j in range(1,10):
#         w[0,j] = weights[0,j-1]
#     for j in range(2,10):
#         w[1,j] = weights[0,7+j]
#     w[2,3] = weights[0,17]
#     w[2,4] = weights[0,18]
#     w[2,5] = weights[0,19]
#     w[3,4] = weights[0,20]
#     w[3,5] = weights[0,21]
#     w[4,5] = weights[0,22]
#     w[6,7] = weights[0,23]
#     w[6,8] = weights[0,24]
#     w[6,9] = weights[0,25]
#     w[7,8] = weights[0,26]
#     w[7,9] = weights[0,27]
#     w[8,9] = weights[0,28]
#     w = w+w.T
#     w = w[::-1].T[::-1]
#     print(w)
# print(weights)
#
# plt.figure()
# ax = plt.gca()
# im = ax.imshow(weightsChangeCompl[0:100,:])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size='5%', pad=0.05)
# plt.colorbar(im, cax = cax, ticks = [-1, 0, 1])
# plt.savefig('WeightsChangeSmallNet.jpg', bbox_inches='tight')
# plt.close()