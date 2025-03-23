from sklearn.preprocessing import StandardScaler

def data_pre_processing(cfg_proj, cfg_m, x_train_raw, y_train, g_train, x_test_raw, y_test, g_test): 

    if "temporal" not in cfg_proj.solver:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)
        return x_train, y_train, g_train, x_test, y_test, g_test
    
    else:
        scaler = StandardScaler()
        X = []
        for sequence in x_train_raw:
            for feature in sequence:
                X.append(feature)
        
        scaler.fit(X)

        for i in range(len(x_train_raw)):
            for j in range(len(x_train_raw[i])):
                x_train_raw[i][j] = scaler.transform(x_train_raw[i][j: j+1])[0]

        for i in range(len(x_test_raw)):
            for j in range(len(x_test_raw[i])):
                x_test_raw[i][j] = scaler.transform(x_test_raw[i][j: j+1])[0]

    return x_train_raw, y_train, g_train, x_test_raw, y_test, g_test