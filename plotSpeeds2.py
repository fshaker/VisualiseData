import numpy as np

import matplotlib.pyplot as plt



files = np.load("E:/Fujitsu/AnomallyDetectionProject/recoveredSpeedMai_2.npz")
recovered=files['recovered']
ActualSpeed = files['ActualSpeed']

p1, = plt.plot(recovered, label='recovered')
#p2, = plt.plot(ActualSpeed, label='Actual')
plt.legend(handles=[p1])
plt.xlabel('Links')

plt.ylabel('speed')
plt.title('Recovered speed for the first 100 links')

plt.savefig('recoveredVSactual_Mai_2_rec.jpg', bbox_inches='tight')
plt.close()