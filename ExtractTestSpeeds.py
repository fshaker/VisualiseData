import tkinter as tk
from tkinter import filedialog
import pickle
import os
import numpy as np
from tempfile import TemporaryFile

root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory() #Open folder test_data_10-31
TestData = np.zeros((22,253,287))

f = open("TrafficData.pkl", "rb")
trafficData = pickle.load(f)
f.close()

TestData[0:15,:,:] = trafficData[28:43, :,:]
i=15
for filename in os.listdir(folder_path)[15:]:
    file_path = folder_path + "/" + filename
    TestData[i, :, :] = pickle.load( open(file_path, "rb"))
    i=i+1

pickle_out = open("TestData.pkl", "wb")
pickle.dump(TestData, pickle_out, protocol=2)
pickle_out.close()


