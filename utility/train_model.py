import sys
import os
from pathlib import Path
import logging
import json
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.heart_model import HeartFailureModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model():
    try:
        # Initialize paths
        data_path = project_root / 'data' / 'processed_data.csv'
        model_path = project_root / 'model' / 'heart_failure_model.joblib'
        
        # Check if processed data exists
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data file not found at {data_path}")
        
        # Initialize and train model
        logging.info("Initializing model training...")
        model = HeartFailureModel()
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = model.load_data(data_path)
        X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
        
        # Train model
        logging.info("Training model with hyperparameter tuning...")
        best_params, best_score = model.train(X_train_scaled, y_train)
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best cross-validation score (ROC AUC): {best_score:.4f}")
        
        # Evaluate model
        logging.info("Evaluating model performance...")
        metrics = model.evaluate(X_test_scaled, y_test)
        
        logging.info("Model Performance Metrics:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"ROC AUC: {metrics['auc']:.4f}")
        logging.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logging.info(f"Specificity: {metrics['specificity']:.4f}")
        
        # Get and plot feature importance
        logging.info("Generating feature importance plot...")
        feature_importance = model.get_feature_importance()
        logging.info("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
        
        # Save model
        logging.info(f"Saving model to {model_path}")
        model.save_model(model_path)
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            'accuracy': float(metrics['accuracy']),
            'auc': float(metrics['auc']),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        
        # Save metrics
        metrics_path = project_root / 'model' / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        logging.info("Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 