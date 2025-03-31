import os
import pandas as pd
from pathlib import Path

def prepare_data():
    """Clean and prepare the Cleveland Heart Disease dataset."""
    # Read the raw data
    data_path = Path("data/raw_data.csv")
    if not data_path.exists():
        raise FileNotFoundError("Raw data file not found. Please make sure 'data/raw_data.csv' exists.")
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Convert numeric columns
    numeric_columns = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
    ]
    
    # Convert categorical columns
    categorical_columns = [
        'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
    ]
    
    # Ensure correct data types
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in categorical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert target to binary (0: no heart disease, 1: heart disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Save processed dataset
    output_path = Path("data/processed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print("\nClass distribution:")
    print(df['target'].value_counts(normalize=True))
    
    return df

if __name__ == "__main__":
    try:
        prepare_data()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure 'data/raw_data.csv' exists")
        print("2. Verify that you have write permissions in the data directory")
        print("3. Check that the data file is in the correct format") 