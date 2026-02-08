import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


file_name="Heart (1).xlsx"
df=pd.read_excel(file_name)

print("dataset loaded successfully")
print(df.head())

df=df.drop(columns=["Unnamed :0"],errors="ignore")

print("\n(a) shape of dataset:")
print(df.shape)

print("\n(b)missing values in each column:")
print(df.isnull().sum())

print("\n(c)Data type of each column:")
print(df.dtypes)

numeric_cols = df.select_dtypes(include=np.number).columns
zero_count = (df[numeric_cols]==0).sum()
print("\n(d) Zero values count in numeric number:")
print(zero_count)

mean_age = df["Age"].mean()
print("\n(e) Mean age of patients:")
print(round(mean_age,2))

X=df[["Age","Sex","ChestPain","RestBP","Chol"]]

X_train, X_test = train_test_split(X,test_size=0.25,random_state=42)

print("\n(f) Extracted columns: Age, Sex, ChestPain, RestBP, Chol")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\n✅ Done!")