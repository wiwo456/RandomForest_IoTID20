# This code is about runnign the 36 features that were run in the reseach paper. Here, F1 score, accuracy, precison all is calculated.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("IoTID20.csv", nrows=100000)
print("Dataset loaded! Shape:", data.shape)

data = data.drop_duplicates()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

remove_cols = ['Src_IP', 'Dst_IP', 'Flow_ID', 'Sub_Cat', 'Timestamp']
for col in remove_cols:
    if col in data.columns:
        data = data.drop(col, axis=1)

encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

features = [
    'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts',
    'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std',
    'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Std',
    'Flow_Byts/s', 'Flow_Pkts/s', 'Flow_IAT_Mean', 'Flow_IAT_Std',
    'Fwd_IAT_Tot', 'Fwd_IAT_Mean', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean',
    'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Fwd_PSH_Flags',
    'FIN_Flag_Cnt', 'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt',
    'Pkt_Size_Avg', 'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts',
    'Active_Mean', 'Active_Std', 'Idle_Mean', 'Idle_Std'
]
features = [f for f in features if f in data.columns]

X = data[features]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

accuracy_lib = accuracy_score(y_test, y_test_pred)
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nRANDOM FOREST (36 FEATURES)")
print("Confusion Matrix:\n", cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy (library): {accuracy_lib*100:.2f}%")
print(f"Accuracy (manual) : {accuracy_manual*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")
