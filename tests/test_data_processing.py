# tests/test_data_preprocessing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the cleaned data
df = pd.read_csv('data/processed/credit_data_clean.csv')

# Create output directory if it doesn't exist
output_dir = 'outputs/preprocessed/'
os.makedirs(output_dir, exist_ok=True)

# Get numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Loop and save boxplots
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    
    # Save plot
    filename = f'{output_dir}boxplot_{col}.png'
    plt.savefig(filename)
    print(f'✅ Saved: {filename}')
    
    plt.close()  # Close the figure to free memory
