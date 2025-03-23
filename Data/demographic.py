import pandas as pd
import pickle
import numpy as np

df = pd.read_csv("rawdata/Baseline_label.csv")

with open("rawdata/id2feature.p", "rb") as f:
    id2feautre = pickle.load(f)

original_subjects = list(id2feautre.keys())


for label in range(3):
    if label < 2:
        subjects = [subject for subject in original_subjects if 1-int(df[df["ts_sub_id"] == subject]['nac_normcog'].values[0]) == label]
    else:
        subjects = original_subjects

    age = [float(df[df["ts_sub_id"] == subject]['nac_a1_age'].values[0]) for subject in subjects]
    gender = 100*len([subject for subject in subjects if int(df[df["ts_sub_id"] == subject]['nac_sex'].values[0]) == 2])/len(subjects)
    edu = [int(df[df["ts_sub_id"] == subject]['nac_educ'].values[0]) for subject in subjects]
    
    numConv = [len(id2feautre[subject]) for subject in subjects]
    
    if label == 0:
        print("NL (n = %d)"%len(subjects))
    elif label == 1:
        print("MCI (n = %d)"%len(subjects))
    else:
        print("All (n = %d)"%len(subjects))
        
    print("Age:", "%.1f"%np.mean(age), "\u00B1", "%.1f"%np.std(age))
    print("Gender (% women):", "%.1f"%gender)
    print("Edu:", "%.1f"%np.mean(edu), "\u00B1", "%.1f"%np.std(edu))
    print("Number of Conversations:", "%.1f"%np.mean(numConv), "\u00B1", "%.1f"%np.std(numConv))