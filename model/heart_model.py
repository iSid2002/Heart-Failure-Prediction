import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class HeartFailureModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
    def load_data(self, data_path):
        """Load and split the dataset."""
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test):
        """Preprocess the data using StandardScaler."""
        # Separate numerical and categorical features
        numerical_features = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
        ]
        categorical_features = [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
        ]
        
        # Scale only numerical features
        X_train_numerical = self.scaler.fit_transform(X_train[numerical_features])
        X_test_numerical = self.scaler.transform(X_test[numerical_features])
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_numerical,
            columns=numerical_features,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_numerical,
            columns=numerical_features,
            index=X_test.index
        )
        
        # Add categorical features back
        X_train_scaled[categorical_features] = X_train[categorical_features]
        X_test_scaled[categorical_features] = X_test[categorical_features]
        
        # Ensure same column order as feature_names
        X_train_scaled = X_train_scaled[self.feature_names]
        X_test_scaled = X_test_scaled[self.feature_names]
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train):
        """Train the model with hyperparameter tuning."""
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': conf_matrix
        }
    
    def get_feature_importance(self):
        """Get and plot feature importance."""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('model/feature_importance.png')
        plt.close()
        
        return feature_importance
    
    def predict(self, X):
        """Make predictions on new data."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        # Scale numerical features
        numerical_features = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
        ]
        categorical_features = [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
        ]
        
        # Scale numerical features
        X_numerical = self.scaler.transform(X[numerical_features])
        
        # Create scaled DataFrame
        X_scaled = pd.DataFrame(
            X_numerical,
            columns=numerical_features,
            index=X.index
        )
        
        # Add categorical features
        X_scaled[categorical_features] = X[categorical_features]
        
        # Ensure correct column order
        X_scaled = X_scaled[self.feature_names]
        
        # Get predictions and probabilities
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)[:, 1]
        
        return prediction, probability
    
    def save_model(self, model_path):
        """Save the trained model and scaler."""
        model_dir = Path(model_path).parent
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_dir / 'scaler.joblib')
    
    def load_model(self, model_path):
        """Load a trained model and scaler."""
        model_dir = Path(model_path).parent
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(model_dir / 'scaler.joblib')