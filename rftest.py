# This a test through random forest algorithm. This code is about runnign the 11 features that I found online and were common in different research papers. They also were some of the important features in the data set.  Here, F1 score, accuracy, precison all is calculated.

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

if 'Label' not in data.columns:
    raise ValueError("Label column not found!")
encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

features = [
    'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts',
    'Flow_Byts/s', 'Flow_Pkts/s',
    'Pkt_Len_Mean', 'Pkt_Len_Std',
    'Flow_IAT_Mean', 'Flow_IAT_Std',
    'Active_Mean', 'Idle_Mean'
]
features = [f for f in features if f in data.columns]

X = data[features]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\nAccuracy Results:")
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy : {test_acc*100:.2f}%")

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nModel Evaluation (from TP/FP/TN/FN):")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")

print("\nModel trained and evaluated successfully using 11 key IoT features.")
