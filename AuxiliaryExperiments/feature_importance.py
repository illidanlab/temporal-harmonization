import pickle
import pandas as pd

# "temporal_sequence_harmonization_adversarial_solver", "temporal_solver"
solver = "temporal_sequence_harmonization_adversarial_solver"
df = pd.read_csv("rawdata/feature_name.csv")
names = list(df["Feature Name"])
with open(f"checkpoints/feature_importance_{solver}.pkl", "rb") as f:
    feature_importance = list(pickle.load(f))

if solver == "temporal_solver":
    auc_real = 0.648125
else:
    auc_real = 0.72078125
    
for i in range(len(feature_importance)):
    feature_importance[i] = [auc_real - feature_importance[i], i]

feature_importance.sort(key = lambda x: -x[0])
feature_importance = feature_importance[:10]

for i in feature_importance:
    if(i[1] < 64):
        t = "LIWC"
    elif(i[1] < 64 + 23):
        t = "Syntactic"
    elif(i[1] < 64 + 23 + 10):
        t = "Lexical Diversity"
    else:
        t = "Response Length"

    print(names[i[1]], " ", t, " ","%.5f"%i[0])