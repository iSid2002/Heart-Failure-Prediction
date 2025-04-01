import pandas as pd
from model.heart_model import HeartFailureModel
import numpy as np

def test_model():
    model = HeartFailureModel()
    
    test_cases = [
        # 1. Extremely Low Risk Case (Young, Very Healthy)
        {
            'name': 'Extremely Low Risk - Young Very Healthy',
            'data': {
                'age': 25,
                'sex': 0,  # Female
                'cp': 0,   # No chest pain
                'trestbps': 105,  # Optimal BP
                'chol': 150,  # Very low cholesterol
                'fbs': 0,   # Normal blood sugar
                'restecg': 0,  # Normal ECG
                'thalach': 185,  # Excellent max heart rate
                'exang': 0,   # No exercise angina
                'oldpeak': 0.0,  # No ST depression
                'slope': 2,   # Upsloping
                'ca': 0,     # No vessels affected
                'thal': 2    # Normal
            }
        },
        
        # 2. Critical Risk Case
        {
            'name': 'Critical Risk - Multiple Severe Conditions',
            'data': {
                'age': 75,
                'sex': 1,  # Male
                'cp': 3,   # Asymptomatic
                'trestbps': 190,  # Very high BP
                'chol': 380,  # Very high cholesterol
                'fbs': 1,   # High blood sugar
                'restecg': 2,  # Abnormal ECG
                'thalach': 95,  # Very low max heart rate
                'exang': 1,   # Exercise angina
                'oldpeak': 5.0,  # Severe ST depression
                'slope': 0,   # Downsloping
                'ca': 3,     # Three vessels affected
                'thal': 1    # Fixed defect
            }
        },
        
        # 3. Borderline Case (Mixed Indicators)
        {
            'name': 'Borderline Case - Mixed Indicators',
            'data': {
                'age': 55,
                'sex': 1,  # Male
                'cp': 1,   # Atypical angina
                'trestbps': 140,  # Borderline BP
                'chol': 240,  # Borderline cholesterol
                'fbs': 0,   # Normal blood sugar
                'restecg': 1,  # ST-T abnormality
                'thalach': 145,  # Average max heart rate
                'exang': 0,   # No exercise angina
                'oldpeak': 1.5,  # Moderate ST depression
                'slope': 1,   # Flat
                'ca': 1,     # One vessel affected
                'thal': 2    # Normal
            }
        },
        
        # 4. Young with Critical Conditions
        {
            'name': 'Young with Critical Conditions',
            'data': {
                'age': 35,
                'sex': 0,  # Female
                'cp': 3,   # Asymptomatic
                'trestbps': 165,  # High BP
                'chol': 310,  # High cholesterol
                'fbs': 1,   # High blood sugar
                'restecg': 2,  # Abnormal ECG
                'thalach': 160,  # Good max heart rate
                'exang': 1,   # Exercise angina
                'oldpeak': 4.8,  # Critical ST depression
                'slope': 0,   # Downsloping
                'ca': 3,     # Three vessels affected
                'thal': 1    # Fixed defect
            }
        },
        
        # 5. Elderly with Protective Factors
        {
            'name': 'Elderly with Protective Factors',
            'data': {
                'age': 68,
                'sex': 1,  # Male
                'cp': 0,   # No chest pain
                'trestbps': 118,  # Normal BP
                'chol': 185,  # Normal cholesterol
                'fbs': 0,   # Normal blood sugar
                'restecg': 0,  # Normal ECG
                'thalach': 155,  # Good max heart rate for age
                'exang': 0,   # No exercise angina
                'oldpeak': 0.2,  # Minimal ST depression
                'slope': 2,   # Upsloping
                'ca': 0,     # No vessels affected
                'thal': 2    # Normal
            }
        },
        
        # 6. Middle-aged with Single Critical Factor
        {
            'name': 'Middle-aged with Single Critical Factor',
            'data': {
                'age': 50,
                'sex': 1,  # Male
                'cp': 3,   # Asymptomatic
                'trestbps': 135,  # Normal BP
                'chol': 220,  # Normal cholesterol
                'fbs': 0,   # Normal blood sugar
                'restecg': 0,  # Normal ECG
                'thalach': 165,  # Good max heart rate
                'exang': 1,   # Exercise angina
                'oldpeak': 4.9,  # Critical ST depression
                'slope': 0,   # Downsloping
                'ca': 0,     # No vessels affected
                'thal': 2    # Normal
            }
        },
        
        # 7. All Minimum Values
        {
            'name': 'Minimum Values',
            'data': {
                'age': 25,
                'sex': 0,
                'cp': 0,
                'trestbps': 90,
                'chol': 120,
                'fbs': 0,
                'restecg': 0,
                'thalach': 70,
                'exang': 0,
                'oldpeak': 0.0,
                'slope': 0,
                'ca': 0,
                'thal': 0
            }
        },
        
        # 8. All Maximum Values
        {
            'name': 'Maximum Values',
            'data': {
                'age': 80,
                'sex': 1,
                'cp': 3,
                'trestbps': 200,
                'chol': 400,
                'fbs': 1,
                'restecg': 2,
                'thalach': 200,
                'exang': 1,
                'oldpeak': 6.2,
                'slope': 2,
                'ca': 3,
                'thal': 2
            }
        },
        
        # 9. Young with Multiple Risk Factors
        {
            'name': 'Young with Multiple Risk Factors',
            'data': {
                'age': 32,
                'sex': 0,  # Female
                'cp': 2,   # Non-anginal pain
                'trestbps': 175,  # High BP
                'chol': 320,  # Very high cholesterol
                'fbs': 1,   # High blood sugar
                'restecg': 1,  # ST-T abnormality
                'thalach': 140,  # Below average max heart rate
                'exang': 1,   # Exercise angina
                'oldpeak': 2.8,  # Significant ST depression
                'slope': 0,   # Downsloping
                'ca': 2,     # Two vessels affected
                'thal': 1    # Fixed defect
            }
        },
        
        # 10. Elderly with Mixed Risk Profile
        {
            'name': 'Elderly with Mixed Risk Profile',
            'data': {
                'age': 72,
                'sex': 1,  # Male
                'cp': 1,   # Atypical angina
                'trestbps': 160,  # High BP
                'chol': 250,  # Borderline cholesterol
                'fbs': 0,   # Normal blood sugar
                'restecg': 1,  # ST-T abnormality
                'thalach': 130,  # Low max heart rate
                'exang': 0,   # No exercise angina
                'oldpeak': 1.8,  # Moderate ST depression
                'slope': 1,   # Flat
                'ca': 1,     # One vessel affected
                'thal': 2    # Normal
            }
        }
    ]
    
    print("\nTesting Heart Disease Prediction Model on Edge Cases")
    print("=" * 50)
    
    results = []
    for case in test_cases:
        try:
            result = model.predict(case['data'])
            results.append({
                'name': case['name'],
                'prediction': 'High Risk' if result['prediction'] == 1 else 'Low Risk',
                'probability': result['probability']
            })
            
            print(f"\nTest Case: {case['name']}")
            print(f"Prediction: {results[-1]['prediction']}")
            print(f"Risk Probability: {results[-1]['probability']:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nError in test case {case['name']}: {str(e)}")
    
    # Analyze results
    print("\nSummary Analysis")
    print("=" * 50)
    
    high_risk_cases = [r for r in results if r['prediction'] == 'High Risk']
    print(f"Total test cases: {len(results)}")
    print(f"High risk predictions: {len(high_risk_cases)}")
    print(f"Low risk predictions: {len(results) - len(high_risk_cases)}")
    
    # Find highest and lowest risk cases
    highest_risk = max(results, key=lambda x: x['probability'])
    lowest_risk = min(results, key=lambda x: x['probability'])
    
    print(f"\nHighest risk case: {highest_risk['name']} (Probability: {highest_risk['probability']:.4f})")
    print(f"Lowest risk case: {lowest_risk['name']} (Probability: {lowest_risk['probability']:.4f})")

if __name__ == "__main__":
    test_model() 