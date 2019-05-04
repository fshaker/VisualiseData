import tkinter as tk
from tkinter import filedialog
import pickle
import os
import numpy as np
from tempfile import TemporaryFile

root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory() #
ground_truth = np.zeros((22,253,287))



i=0
for filename in os.listdir(folder_path):
    file_path = folder_path + "/" + filename
    ground_truth[i,:,:]= pickle.load( open(file_path, "rb"))
    i=i+1

pickle_out = open("ground_truth.pickle", "wb")
pickle.dump(ground_truth, pickle_out, protocol=2)
pickle_out.close()


