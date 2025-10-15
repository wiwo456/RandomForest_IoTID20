

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("🚀 Starting Random Forest training...")


df = pd.read_csv("IoTID20.csv")
print("✅ Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

#
df = df.drop_duplicates()
df = df.fillna(0)
print("✅ Cleaned missing values and duplicates.")


encoder = LabelEncoder()
df["Label"] = encoder.fit_transform(df["Label"])
print("✅ Encoded the 'Label' column.")


df = df.select_dtypes(include=['number'])
print("✅ Kept only numeric columns. Shape now:", df.shape)

df = df.replace([np.inf, -np.inf], np.nan)  
df = df.fillna(0)                            
print("✅ Replaced infinity and NaN values.")


X = df.drop(columns=["Label"])
y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("✅ Data split complete.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete!")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", round(accuracy * 100, 2), "%")


print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
print("\n🔎 Sample Prediction ->", prediction)
print("Actual Label ->", y_test.iloc[0])
