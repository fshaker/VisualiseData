import tkinter as tk
from tkinter import filedialog
import pickle
import os
import numpy as np
from tempfile import TemporaryFile

root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory() #first open folder 3_week_... it is from Dec 13 to Jan10
data=np.zeros((43,253,287))#we have 43 days of data
i=0
for filename in os.listdir(folder_path):
    file_path = folder_path + "/" + filename
    data[i,:,:]= pickle.load( open(file_path, "rb"))
    i=i+1
folder_path = filedialog.askdirectory()#Then open folder 1_week_....This is from Dec 25 to Jan24

folderContents = os.listdir(folder_path)
for filename in folderContents[17:]:
    file_path = folder_path + "/" + filename
    data[i,:,:]= pickle.load( open(file_path, "rb"))
    i=i+1
with open('TrafficData.pkl','wb') as f:
    pickle.dump(data, f, protocol=2)
