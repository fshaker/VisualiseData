import pickle

with open("E:\Fujitsu\AnomallyDetectionProject\\test_week_speed_Jan023.pickle", "rb") as f:
    w = pickle.load(f)
pickle.dump(w, open("test_week_speed_Jan023.pickle","wb"), protocol=2)