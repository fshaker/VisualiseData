import numpy as np
import matplotlib.pyplot as plt

# recovered =[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,
#        0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#        0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
#        0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
#        1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
#        0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
#        1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#        0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
#        1, 0]
#
# ActualSpeed = [ True,  True,  True, False,  True, False,  True, False,  True,
#        False, False,  True,  True, False,  True,  True,  True,  True,
#         True,  True,  True, False,  True, False,  True, False,  True,
#        False,  True, False,  True, False,  True, False,  True, False,
#         True, False,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True, False,  True, False,  True, False,  True,
#        False,  True, False,  True, False,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#        False,  True, False,  True, False, False,  True, False, False,
#         True, False, False,  True,  True, False,  True, False,  True,
#        False,  True,  True,  True,  True,  True, False,  True, False,
#         True,  True,  True,  True,  True, False,  True, False,  True,
#         True,  True, False,  True, False,  True, False,  True, False,
#         True,  True,  True, False,  True, False,  True, False,  True,
#        False,  True, False,  True,  True,  True, False,  True, False,
#         True, False,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True, False,  True, False,
#         True, False,  True,  True,  True,  True,  True,  True,  True,
#         True,  True, False,  True,  True,  True, False,  True,  True,
#         True, False,  True, False,  True, False,  True, False,  True,
#        False,  True,  True,  True, False,  True, False,  True, False,
#         True, False]
files = np.load("recoveredSpeed_4state_1.npz")
recovered=files['recovered']
ActualSpeed = files['ActualSpeed']

recovered_num = np.zeros(int(len(recovered)/2),int)
print(int(len(recovered)/2))
print(recovered_num.shape)
for i in range(0,len(recovered),2):
    if recovered[i]==0:
        if recovered[i+1]==0:
            recovered_num[int(i/2)]=0
        else:
            recovered_num[int(i/2)]=1
    else:
        if recovered[i+1]==0:
            recovered_num[int(i/2)]=2
        else:
            recovered_num[int(i/2)]=3

ActualSpeed_num = np.zeros(int(len(ActualSpeed)/2))
for i in range(0,len(ActualSpeed),2):
    if ActualSpeed[i]==0:
        if ActualSpeed[i+1]==0:
            ActualSpeed_num[int(i/2)]=0
        else:
            ActualSpeed_num[int(i/2)]=1
    else:
        if ActualSpeed[i+1]==0:
            ActualSpeed_num[int(i/2)]=2
        else:
            ActualSpeed_num[int(i/2)]=3


# plt.plot(recovered_num)
# plt.plot(ActualSpeed_num)
# plt.show()

p1, = plt.plot(recovered_num, label='recovered')
p2, = plt.plot(ActualSpeed_num, label='Actual')
plt.legend(handles=[p1,p2])
plt.xlabel('Links')

plt.ylabel('speed')
plt.title('Recovered speed vs actual test speed for the first 100 links')
plt.show()
plt.savefig('recoveredVSactual.png')
plt.close()