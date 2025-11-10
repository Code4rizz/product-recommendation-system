import re
import pandas as pd
import numpy as np

data=pd.read_csv('DMart.csv')
print(f"Original dataset: {data.shape[0]} products")

data = data[['Name', 'Price', 'Category', 'SubCategory', 'Description']]


data = data.dropna(subset=['Name', 'Description']) 
data = data[data['Description'].str.strip() != ''] 
data = data.dropna()

data = data.reset_index(drop=True)

print(f"Cleaned dataset: {data.shape[0]} products")
print(f"Removed: {5189 - data.shape[0]} products")

print("\nRemaining null values:")
print(data.isnull().sum())

data.to_csv('DMart_cleaned.csv', index=False)
print("\nCleaned data saved with only necessary columns!")
