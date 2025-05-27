"""
This module provides a model training framework for the BAQ (Bangkok Air Quality) application.
It implements a ModelTrainer class that handles:

1. Model initialization based on configuration parameters
2. Cross-validation with multiple evaluation metrics
3. Model training and evaluation
4. Model persistence and artifact management

The trainer supports multiple model types (Random Forest, SVM, Logistic Regression)
and can be extended to support additional algorithms. It follows a configuration-driven
approach where model selection, hyperparameters, and training settings are defined
in a central configuration file.

This implementation serves as a reference for how model training should be structured
in the BAQ application, ensuring consistency, reproducibility, and proper evaluation
of predictive models.

"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.artifacts_config = config["artifacts"]
        self.model = None
    
    def _get_model(self) -> BaseEstimator:
        """Initialize model based on configuration"""
        model_type = self.model_config["type"]
        hyperparams = self.model_config["hyperparameters"]
        
        if model_type == "random_forest":
            return RandomForestClassifier(**hyperparams["random_forest"])
        elif model_type == "svm":
            return SVC(**hyperparams["svm"], probability=True)
        elif model_type == "logistic_regression":
            return LogisticRegression(**hyperparams["logistic_regression"])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_scoring_metrics(self) -> Dict[str, Any]:
        """Define scoring metrics for multiclass classification"""
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        return scoring
    
    def cross_validate(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation if enabled in config"""
        cv_config = self.training_config["cross_validation"]
        
        if not cv_config.get("enable", False):
            return {}
        
        model = self._get_model()
        metrics = self.training_config["metrics"]
        results = {}
        
        # Use StratifiedKFold for classification tasks
        cv = StratifiedKFold(
            n_splits=cv_config["n_splits"],
            shuffle=cv_config.get("shuffle", True),
            random_state=self.config["data"]["split"]["random_state"]
        )
        
        # Get appropriate scoring metrics for multiclass
        scoring = self._get_scoring_metrics()
        
        for metric in metrics:
            if metric in scoring:
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring=scoring[metric]
                )
                results[f"cv_{metric}_mean"] = cv_scores.mean()
                results[f"cv_{metric}_std"] = cv_scores.std()
        
        return results
    
    def train(self, X_train: np.ndarray, y_train: pd.Series) -> BaseEstimator:
        """Train the model on training data"""
        self.model = self._get_model()
        
        # Ensure y is in the correct format - target should already be encoded
        # but some sklearn models are picky about type
        if isinstance(y_train, pd.Series):
            y_train_values = y_train.values
        else:
            y_train_values = y_train
            
        self.model.fit(X_train, y_train_values)
        return self.model
    
    def save_model(self) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        base_path = self.artifacts_config["base_path"]
        model_path = self.artifacts_config["model"]["path"]
        model_filename = self.artifacts_config["model"]["filename"]
        
        # Create directory if it doesn't exist
        full_path = os.path.join(base_path, model_path)
        os.makedirs(full_path, exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, os.path.join(full_path, model_filename))
        
    def get_feature_importance(self, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Get feature importance if available in the model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Check if model has feature_importances_ attribute (e.g. Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # Check if model has coef_ attribute (e.g. logistic regression)
        elif hasattr(self.model, 'coef_'):
            # For multiclass, take mean of absolute coefficients across classes
            if len(self.model.coef_.shape) > 1 and self.model.coef_.shape[0] > 1:
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
                
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        return None 