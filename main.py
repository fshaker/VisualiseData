import tkinter as tk
from tkinter import filedialog
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.Inf)

root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory()
plt.close("all")
data=np.zeros((253,287,31))
i=0
for filename in os.listdir(folder_path):
    file_path = folder_path + "/" + filename
    data[:,:,i]= pickle.load( open(file_path, "rb"))
    i=i+1
# for i in range(20):
#     for j in range(100,110):
#         plt.ylim(10,100)
#         plt.xlim(1,31)
#         plt.plot(data[i,j,:])
#         plt.show()
print(data[:,:,1])

meanSpeed = np.mean(data,2)
stdSpeed = np.std(data,2)

# for i in range(253):
#     plt.ylim(0, 25)
#     plt.plot(stdSpeed[i,:])
#     plt.show()
