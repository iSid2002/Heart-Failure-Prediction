import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_heart_disease_dataset():
    """Create a realistic heart disease dataset based on medical knowledge"""
    n_samples = 3000  # Further increased sample size
    np.random.seed(42)
    
    # Generate age with a realistic distribution
    age = np.random.normal(55, 12, n_samples).clip(25, 80).astype(int)
    
    # Generate other features with realistic correlations
    data = {
        'age': age,
        'sex': np.random.binomial(1, 0.5, n_samples),  # 0: Female, 1: Male
        'cp': np.random.randint(0, 4, n_samples),      # 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
        'trestbps': np.random.normal(130, 15, n_samples).clip(90, 200).astype(int),  # Resting BP
        'chol': np.random.normal(220, 40, n_samples).clip(120, 400).astype(int),     # Cholesterol
        'fbs': (np.random.normal(100, 20, n_samples) > 120).astype(int),             # Fasting Blood Sugar
        'restecg': np.random.randint(0, 3, n_samples), # Resting ECG
        'thalach': np.random.normal(150, 20, n_samples).clip(70, 200).astype(int),   # Max Heart Rate
        'exang': np.random.binomial(1, 0.3, n_samples), # Exercise Induced Angina
        'oldpeak': np.random.exponential(1, n_samples).clip(0, 6.2),                 # ST Depression
        'slope': np.random.randint(0, 3, n_samples),    # ST Slope
        'ca': np.random.randint(0, 4, n_samples),       # Number of Vessels
        'thal': np.random.randint(0, 3, n_samples)      # Thalassemia
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable using refined medical rules
    y = np.zeros(n_samples)
    
    # Critical risk factors (immediate high risk)
    critical_risk = (
        (df['ca'] >= 3) |                                           # Three or more vessels affected
        (df['oldpeak'] > 4.5) |                                     # Severe ST depression
        ((df['cp'] == 3) & (df['exang'] == 1) & (df['ca'] >= 2))   # Severe angina with multiple vessels
    )
    
    # Severe risk factors (very high risk)
    severe_risk = (
        ((df['age'] > 65) & (df['chol'] > 320)) |                    # Very high age and cholesterol
        ((df['thalach'] < 100) & (df['age'] > 60)) |                 # Very low heart rate in elderly
        ((df['trestbps'] > 180) & (df['chol'] > 300)) |             # Very high BP and cholesterol
        ((df['cp'] == 3) & (df['oldpeak'] > 3))                     # Asymptomatic pain with significant ST depression
    )
    
    # Moderate risk factors
    moderate_risk = (
        ((df['age'] > 60) & (df['chol'] > 280)) |                    # High age and cholesterol
        ((df['cp'] >= 2) & (df['exang'] == 1)) |                     # Chest pain with exercise angina
        (df['oldpeak'] > 2) |                                        # Significant ST depression
        (df['ca'] == 2) |                                           # Two vessels affected
        ((df['thalach'] < 120) & (df['age'] > 50)) |                # Low max heart rate in older adults
        ((df['trestbps'] > 160) & (df['chol'] > 260))              # High BP and cholesterol
    )
    
    # Strong protective factors (low risk indicators)
    protective_factors = (
        (df['age'] < 40) & 
        (df['chol'] < 200) & 
        (df['trestbps'] < 120) & 
        (df['thalach'] > 150) & 
        (df['ca'] == 0) & 
        (df['oldpeak'] < 0.5) &
        (df['cp'] == 0) &                # No chest pain
        (df['exang'] == 0)               # No exercise-induced angina
    )
    
    # Very low risk factors (extremely healthy indicators)
    very_low_risk = (
        (df['age'] < 35) & 
        (df['chol'] < 180) & 
        (df['trestbps'] < 110) & 
        (df['thalach'] > 170) & 
        (df['ca'] == 0) & 
        (df['oldpeak'] == 0) &
        (df['cp'] == 0) &
        (df['exang'] == 0) &
        (df['fbs'] == 0) &
        (df['restecg'] == 0)
    )
    
    # Assign risk levels with priority
    y[critical_risk] = 1
    y[severe_risk & ~critical_risk] = 1
    y[moderate_risk & ~critical_risk & ~severe_risk & ~protective_factors] = 1
    y[protective_factors & ~critical_risk & ~severe_risk] = 0
    y[very_low_risk] = 0  # Override everything except critical risk
    
    return df, y

def train_and_save_model():
    # Create dataset
    print("Creating dataset...")
    X, y = create_heart_disease_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model with refined parameters
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=300,      # Further increased number of trees
        max_depth=15,          # Increased depth for more complex patterns
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',  # Better handling of imbalanced cases
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'heart_failure_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Model saved successfully at {model_path}")
    print(f"Scaler saved successfully at {scaler_path}")
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Test with known cases
    test_cases = [
        ("Very Low Risk", {
            'age': 35,
            'sex': 0,
            'cp': 0,
            'trestbps': 110,
            'chol': 170,
            'fbs': 0,
            'restecg': 0,
            'thalach': 170,
            'exang': 0,
            'oldpeak': 0.0,
            'slope': 2,
            'ca': 0,
            'thal': 2
        }),
        ("Very High Risk", {
            'age': 70,
            'sex': 1,
            'cp': 3,
            'trestbps': 185,
            'chol': 340,
            'fbs': 1,
            'restecg': 2,
            'thalach': 95,
            'exang': 1,
            'oldpeak': 4.5,
            'slope': 0,
            'ca': 3,
            'thal': 1
        })
    ]
    
    print("\nValidation on extreme cases:")
    for name, case in test_cases:
        case_df = pd.DataFrame([case])
        case_scaled = scaler.transform(case_df)
        prob = model.predict_proba(case_scaled)[0][1]
        print(f"{name}: {prob:.4f} risk probability")

if __name__ == "__main__":
    train_and_save_model() 