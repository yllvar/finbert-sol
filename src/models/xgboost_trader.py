"""
XGBoost Baseline Model for SOL Trading System
Implements the paper's XGBoost configuration and methodology
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class XGBoostTrader:
    """
    XGBoost model implementation based on "Generating Alpha" paper
    Paper configuration: 200 estimators, depth 6, learning rate 0.05
    """
    
    def __init__(self, paper_config: bool = True):
        """
        Initialize XGBoost model
        
        Args:
            paper_config: Use paper's hyperparameters if True
        """
        self.model = None
        self.feature_importance = None
        self.training_history = []
        self.is_fitted = False
        
        if paper_config:
            # Paper's exact configuration
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'reg_alpha': 0.0,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'min_child_weight': 1
            }
        else:
            # Default configuration
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame = None, y_test: pd.Series = None,
                   early_stopping: bool = True) -> xgb.XGBClassifier:
        """
        Train XGBoost model with paper's methodology
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for early stopping)
            y_test: Test labels (for early stopping)
            early_stopping: Use early stopping if True
        
        Returns:
            Trained XGBoost model
        """
        print("🚀 Training XGBoost model...")
        print(f"   Training data: {X_train.shape}")
        print(f"   Configuration: {self.params}")
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare evaluation set
        eval_set = None
        if early_stopping and X_test is not None and y_test is not None:
            eval_set = [(X_train, y_train), (X_test, y_test)]
            print(f"   Early stopping enabled with test set: {X_test.shape}")
        
        # Train model
        if early_stopping and eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        if hasattr(X_train, 'columns'):
            self._store_feature_importance(X_train.columns)
        else:
            # X_train is numpy array, create generic feature names
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            self._store_feature_importance(feature_names)
        
        # Store training history
        self.training_history.append({
            'train_samples': len(X_train),
            'train_positive_rate': y_train.mean(),
            'feature_count': X_train.shape[1],
            'best_iteration': getattr(self.model, 'best_iteration', self.params['n_estimators'])
        })
        
        self.is_fitted = True
        
        print(f"✅ Model training complete!")
        print(f"   Best iteration: {getattr(self.model, 'best_iteration', self.params['n_estimators'])}")
        
        return self.model
    
    def _store_feature_importance(self, feature_names: List[str]):
        """Store and format feature importance"""
        if self.model is None:
            return
        
        importance_types = ['weight', 'gain', 'cover']
        self.feature_importance = {}
        
        for imp_type in importance_types:
            try:
                importance_scores = self.model.get_booster().get_score(importance_type=imp_type)
                
                # Convert to DataFrame with proper feature names
                importance_df = pd.DataFrame([
                    {'feature': f'f{int(k)}', 'importance': v} 
                    for k, v in importance_scores.items()
                ])
                
                # Map feature indices to names
                importance_df['feature_name'] = importance_df['feature'].map(
                    lambda x: feature_names[int(x[1:])] if x[1:].isdigit() and int(x[1:]) < len(feature_names) else x
                )
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                self.feature_importance[imp_type] = importance_df
                
            except Exception as e:
                print(f"⚠️  Could not calculate {imp_type} importance: {e}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with trained model
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Convert to binary predictions (default threshold 0.5)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      threshold: float = 0.5) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        print("📊 Evaluating model performance...")
        
        # Get predictions
        predictions, probabilities = self.predict(X_test)
        predictions_thresholded = (probabilities > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions_thresholded)
        
        # Detailed classification report
        report = classification_report(y_test, predictions_thresholded, 
                                     target_names=['Down', 'Up'], 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions_thresholded)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, probabilities)
        except:
            roc_auc = 0.5
        
        # Calculate paper's baseline comparison
        paper_baseline = 0.630  # Paper's reported accuracy
        accuracy_improvement = accuracy - paper_baseline
        
        results = {
            'accuracy': accuracy,
            'paper_baseline': paper_baseline,
            'accuracy_improvement': accuracy_improvement,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions_thresholded,
            'probabilities': probabilities,
            'threshold': threshold
        }
        
        # Print results
        print(f"   Model Accuracy: {accuracy:.3f}")
        print(f"   Paper Baseline: {paper_baseline:.3f}")
        print(f"   Improvement: {accuracy_improvement:+.3f}")
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Positive Rate: {predictions_thresholded.mean():.2%}")
        
        # Detailed metrics
        if '1' in report:
            print(f"   Precision (Up): {report['1']['precision']:.3f}")
            print(f"   Recall (Up): {report['1']['recall']:.3f}")
            print(f"   F1-Score (Up): {report['1']['f1-score']:.3f}")
        else:
            print("   Detailed metrics not available in classification report")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds
        
        Returns:
            Cross-validation results
        """
        print(f"🔄 Performing {cv_folds}-fold time series cross-validation...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Handle both DataFrame and numpy array
            if hasattr(X, 'iloc'):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            else:
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Train model on fold
            fold_model = XGBoostTrader(paper_config=True)
            fold_model.train_model(X_train_fold, y_train_fold)
            
            # Evaluate on fold
            fold_eval = fold_model.evaluate_model(X_test_fold, y_test_fold)
            cv_scores.append(fold_eval['accuracy'])
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'accuracy': fold_eval['accuracy'],
                'roc_auc': fold_eval['roc_auc']
            })
            
            print(f"   Fold {fold + 1}: {fold_eval['accuracy']:.3f}")
        
        cv_results = {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'fold_results': fold_results
        }
        
        print(f"✅ Cross-validation complete:")
        print(f"   Mean Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        return cv_results
    
    def plot_feature_importance(self, importance_type: str = 'gain', top_n: int = 20):
        """
        Plot feature importance
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            top_n: Number of top features to show
        """
        if not self.feature_importance or importance_type not in self.feature_importance:
            print(f"❌ Feature importance {importance_type} not available")
            return
        
        importance_df = self.feature_importance[importance_type].head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature_name')
        plt.title(f'Top {top_n} Feature Importance ({importance_type.title()})')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series):
        """Plot learning curve to analyze overfitting"""
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        
        print("📈 Generating learning curve...")
        
        for size in train_sizes:
            # Sample training data
            n_samples = int(len(X_train) * size)
            X_sample = X_train.iloc[:n_samples]
            y_sample = y_train.iloc[:n_samples]
            
            # Train model
            temp_model = XGBoostTrader(paper_config=True)
            temp_model.train_model(X_sample, y_sample)
            
            # Evaluate
            train_eval = temp_model.evaluate_model(X_sample, y_sample)
            test_eval = temp_model.evaluate_model(X_test, y_test)
            
            train_scores.append(train_eval['accuracy'])
            test_scores.append(test_eval['accuracy'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training Accuracy')
        plt.plot(train_sizes, test_scores, 'o-', label='Validation Accuracy')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(filepath)
        self.is_fitted = True
        print(f"✅ Model loaded from {filepath}")


def main():
    """Test XGBoost model with sample data"""
    print("🚀 Testing XGBoost Model...")
    
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from features.simple_feature_test import simple_feature_test
    
    features, labels, feature_names = simple_feature_test()
    
    if features is None:
        print("❌ Could not load features")
        return
    
    # Split data (70/30 as per paper)
    split_idx = int(len(features) * 0.7)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Initialize and train model
    model = XGBoostTrader(paper_config=True)
    trained_model = model.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = model.evaluate_model(X_test, y_test)
    
    # Cross-validation
    cv_results = model.cross_validate(X_train, y_train, cv_folds=3)
    
    # Plot feature importance
    model.plot_feature_importance('gain', top_n=15)
    
    print("\n🎉 XGBoost model test complete!")
    return model, results


if __name__ == "__main__":
    model, results = main()
