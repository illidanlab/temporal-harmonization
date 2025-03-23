import pandas as pd
import numpy as np
import pickle
import random
import copy

# Load Data 
def load_raw_data(cfg_proj = None):    
    with open("rawdata/id2feature.p", "rb") as f:
        dic_id2feature = pickle.load(f)

    for id in dic_id2feature:
        dic_id2feature[id] = np.array(dic_id2feature[id])
    

    df_labels = pd.read_csv("rawdata/Baseline_label.csv")

    nl_subject = df_labels["ts_sub_id"][df_labels["nac_normcog"] == 1].to_list()
    mci_subject = df_labels["ts_sub_id"][df_labels["nac_normcog"] == 0].to_list()
    nl_subject = [subject for subject in nl_subject if subject in dic_id2feature]
    mci_subject = [subject for subject in mci_subject if subject in dic_id2feature]

    return dic_id2feature, df_labels, nl_subject, mci_subject

# Get Feature from Train and Test Subject ID
def get_feature_from_id(train, test, dic_id2feature, df_labels, sequence = False):
    x_train, y_train, g_train, x_test, y_test, g_test  = [], [], [], [], [], []
    
    for id in train:
        label = 1-int(df_labels[df_labels["ts_sub_id"] == id]['nac_normcog'].values[0])        
        sequences = []
        for feature in dic_id2feature[id]:
            sequences.append(feature)
            if not sequence:
                x_train.append(np.array(feature))
                y_train.append(label*1.0)
                g_train.append(id)

        if sequence:
            x_train.append(np.array(sequences))
            y_train.append(label*1.0)
            g_train.append(id)
            
    for id in test:
        label = 1-int(df_labels[df_labels["ts_sub_id"] == id]['nac_normcog'].values[0])
        sequences = []
        for feature in dic_id2feature[id]:
            if not sequence:
                x_test.append(feature)
                y_test.append(label)
                g_test.append(id)
            sequences.append(feature)
            
        if sequence:
            x_test.append(np.array(sequences))
            y_test.append(label)
            g_test.append(id)
            
    return x_train, np.array(y_train), g_train, x_test, np.array(y_test), g_test

# Premute Feature for Fetaure Importance Experiment
def permute_feature(x_test, index = -1):
    x_test_permute = copy.deepcopy(x_test)
    
    feature_values = []

    for i in x_test:
        for j in i:
            feature_values.append(j[index])
    
    random.shuffle(feature_values)

    current = 0

    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            x_test_permute[i][j][index] = feature_values[current]
            current += 1

    return x_test_permute
