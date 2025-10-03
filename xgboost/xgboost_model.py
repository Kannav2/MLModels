"""
XGBoost Model Implementation
============================

This module defines an XGBoost model that can perform both classification and regression
with commonly used hyperparameters. XGBoost is an optimized gradient boosting framework 
designed for speed and performance.

Usage:
    - For classification: XGBoostModel(is_classifier=True)
    - For regression: XGBoostModel(is_classifier=False)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time



class XGBoostModel:
    def __init__(self, verbose=True, use_class_weights=False, is_classifier=True):
        """
        Initialize XGBoost model with configurable verbosity, class weights, and task type.
        
        Args:
            verbose (bool): Whether to show training progress and detailed output
            use_class_weights (bool): Whether to use balanced class weights (only for classification)
            is_classifier (bool): Whether to use classification (True) or regression (False)
        """
        self.verbose = verbose
        self.use_class_weights = use_class_weights
        self.is_classifier = is_classifier
        
        # Base parameters with detailed explanations , again these are all for my reference, might remove this in production base code
        params = {
            'n_estimators': 100,           # Number of gradient boosted trees - balance between performance and overfitting
            'max_depth': 6,                # Maximum tree depth for base learners - controls model complexity
            'learning_rate': 0.1,          # Boosting learning rate (eta) - smaller values need more trees but reduce overfitting
            'subsample': 0.8,              # Subsample ratio of training instances - prevents overfitting by using random samples
            'colsample_bytree': 0.8,       # Subsample ratio of columns when constructing each tree - feature bagging for diversity
            'random_state': 42,            # Random seed for reproducibility - ensures consistent results across runs
            'n_jobs': -1,                  # Use all available cores for parallel processing - speeds up training
            'early_stopping_rounds': 10   # Stop training if no improvement for 10 rounds - prevents overfitting
        }
        
        # Set task-specific parameters
        if self.is_classifier:
            params['eval_metric'] = 'mlogloss'  # Multi-class log loss for validation data - appropriate for multi-class classification
            # Add class weights if specified (only for classification)
            if use_class_weights:
                from sklearn.utils.class_weight import compute_class_weight
                # Will compute weights after seeing the data
                params['objective'] = 'multi:softprob'
            self.model = xgb.XGBClassifier(**params)
        else:
            params['eval_metric'] = 'rmse'  # Root mean square error for regression
            self.model = xgb.XGBRegressor(**params)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    
    def preprocess_data(self, dataset_path, target):
        """
        Load and preprocess the dataset, splitting it into train/validation/test sets.
        
        Args:
            dataset_path (str): Path to the CSV dataset
            target (str): Name of the target column
        """
        if self.verbose:
            print(f"Loading dataset from: {dataset_path}")
            print(f"Target column: {target}")
        
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        if self.verbose:
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        
        # Check if target column exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Separate features and target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        
        # Handle categorical columns - convert object columns to numeric
        # Remove source_employer as it's not needed for prediction
        if 'source_employer' in self.X.columns:
            self.X = self.X.drop(columns=['source_employer'])
        
        # For regression models, remove specific columns that are not needed
        if not self.is_classifier:
            columns_to_remove = [
                'Total Lost Days (Modified Days Removed)',
                'Calendar Days Total Lost Days (Modified Days Removed)',
                'Total Lost Days',
                'Calendar Days Total Lost Days',
                'Duration in Weeks (Calendar Days)',
                'Weeks Open'
            ]
            
            # Remove columns that exist in the dataset
            existing_columns_to_remove = [col for col in columns_to_remove if col in self.X.columns]
            if existing_columns_to_remove:
                if self.verbose:
                    print(f"Removing {len(existing_columns_to_remove)} columns for regression: {existing_columns_to_remove}")
                self.X = self.X.drop(columns=existing_columns_to_remove)
        
        # Convert any remaining object columns to numeric (one-hot encoding)
        object_columns = self.X.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            if self.verbose:
                print(f"Converting {len(object_columns)} object columns to numeric...")
            self.X = pd.get_dummies(self.X, columns=object_columns, drop_first=True)
        
        if self.verbose:
            print(f"Features shape: {self.X.shape}")
            print(f"Target distribution:\n{self.y.value_counts()}")
        
        # Split data: 60% train, 20% validation, 20% test
        # Use stratification only for classification tasks
        if self.is_classifier:
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.4, random_state=42, stratify=self.y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        else:
            # For regression, no stratification needed
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=0.4, random_state=42
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        
        # Store the splits
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        if self.verbose:
            print(f"Data split completed:")
            print(f"  Training set: {X_train.shape[0]} samples")
            print(f"  Validation set: {X_val.shape[0]} samples")
            print(f"  Test set: {X_test.shape[0]} samples")

    def return_weights(self):
        # Aggressive manual weights for higher recall
        weight_map = {0: 1, 1: 8, 2: 20}  # Very aggressive for class 2
        sample_weights = np.array([weight_map[i] for i in self.y_train])
        return sample_weights
    
    def train_model(self):
        """
        Train the XGBoost model with validation data and early stopping.
        
        Returns:
            Trained model
        """
        if self.verbose:
            print("\n" + "="*50)
            print(f"Starting model training{'with class weights' if self.use_class_weights else ''}...")
            print("="*50)
        
        # Compute class weights if needed
        if self.use_class_weights:

            sample_weights = self.return_weights()
        
        # Record training start time
        start_time = time.time()
        
        # Train with validation set for early stopping and progress monitoring
        if self.use_class_weights:
            self.model.fit(
                self.X_train, self.y_train,
                sample_weight=sample_weights,
                eval_set=[(self.X_val, self.y_val)],
                verbose=self.verbose
            )
        else:
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=self.verbose
            )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nModel training completed in {training_time:.2f} seconds")
            print(f"Best iteration: {self.model.best_iteration}")
            print(f"Best score: {self.model.best_score}")
        
        return self.model

    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.verbose:
            print("\n" + "="*50)
            print("Evaluating model on test data...")
            print("="*50)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        if self.is_classifier:
            # Classification metrics
            accuracy = accuracy_score(self.y_test, predictions)
            
            # Generate detailed results
            results = {
                'accuracy': accuracy,
                'classification_report': classification_report(self.y_test, predictions),
                'confusion_matrix': confusion_matrix(self.y_test, predictions)
            }
            
            # Print results
            print(f"\nModel Evaluation Results (Classification):")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, predictions))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(self.y_test, predictions))
        else:
            # Regression metrics
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            
            # Generate detailed results
            results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            # Print results
            print(f"\nModel Evaluation Results (Regression):")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
        
        # Feature importance display (works for both classification and regression)
        try:
            importances = getattr(self.model, 'feature_importances_', None)
            if importances is not None and self.X_train is not None:
                fi = pd.Series(importances, index=self.X_train.columns).sort_values(ascending=False)
                top_n = 15 if len(fi) > 15 else len(fi)
                print(f"\nTop {top_n} Feature Importances:")
                print(fi.head(top_n))
        except Exception as _:
            pass
        
        return results

    def tune_hyperparameters(self, type="grid_search",grid_size='small',use_class_weights=False):
        """
        Simple hyperparameter tuning using grid search.
        
        Args:
            type (str): 'grid_search' or 'random_search'
            grid_size (str): 'small' or 'medium' parameter grid this is only used for grid search
            
        Returns:
            Dictionary with best parameters and score
        """
        if self.verbose:
            print(f"\nStarting hyperparameter tuning with {grid_size} grid...")
        
        # Define parameter grids
        param_grids = {
            'small': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'medium': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            }
        }

        random_search_params = {
            'n_estimators': np.arange(50, 300, 50),
            'max_depth': np.arange(3, 10, 2),
            'learning_rate': np.arange(0.01, 0.2, 0.05),
            'subsample': np.arange(0.7, 1.0, 0.1),
            'colsample_bytree': np.arange(0.7, 1.0, 0.1)
        }
        # Get parameter grid
        
        # Create base model
        if self.is_classifier:
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            scoring = 'accuracy'
        else:
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            scoring = 'neg_mean_squared_error'
        
        # Get sample weights if needed (only for final training, not for hyperparameter search)
        if use_class_weights and self.is_classifier:
            sample_weights = self.return_weights()
        else:
            sample_weights = None
        

        
        
        # Perform grid search
        start_time = time.time()

        if type == "grid_search":
            param_grid = param_grids[grid_size]
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            best_params = grid_search.best_params_
        
        else:
            random_search = RandomizedSearchCV(
                base_model, random_search_params, cv=3, scoring=scoring, n_jobs=-1, verbose=1
            )
            random_search.fit(self.X_train, self.y_train)
            best_params = random_search.best_params_
        
        
        
        # Create new model with best parameters
        if self.is_classifier:
            self.model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        else:
            self.model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        
        # Retrain with validation set for early stopping
        if sample_weights is not None:
            self.model.fit(
                self.X_train, self.y_train,
                sample_weight=sample_weights,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
        else:
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
        
        training_time = time.time() - start_time
        
        if self.verbose and type == "grid_search":
            print(f"\nTuning completed in {training_time:.2f} seconds")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {grid_search.best_score_:.4f}")

        elif self.verbose and type == "random_search":
            print(f"\nTuning completed in {training_time:.2f} seconds")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {random_search.best_score_:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': grid_search.best_score_ if type == "grid_search" else random_search.best_score_,
            'training_time': training_time
        }


if __name__ == "__main__":
    dataset_path = '/Users/kannavsethi/AcclaimProjects/MLModels/data/data_xgboost_combined.csv'
    target_regression = 'Duration in Weeks (Business Days)'
    target_classification = 'Claim Decision'
    
    # Model 1: Classification without class weights
    print("=" * 80)
    print("MODEL 1: CLASSIFICATION WITHOUT CLASS WEIGHTS")
    print("=" * 80)
    model1 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=True)
    model1.preprocess_data(dataset_path, target_classification)
    model1.train_model()
    model1.evaluate_model()
    
    # Model 2: Classification with aggressive class weights
    print("\n" + "=" * 80)
    print("MODEL 2: CLASSIFICATION WITH AGGRESSIVE CLASS WEIGHTS")
    print("=" * 80)
    model2 = XGBoostModel(verbose=True, use_class_weights=True, is_classifier=True)
    model2.preprocess_data(dataset_path, target_classification)
    model2.train_model()
    model2.evaluate_model()
    
    # Model 3: Regression (continuous value prediction)
    print("\n" + "=" * 80)
    print("MODEL 3: REGRESSION FOR CONTINUOUS VALUES")
    print("=" * 80)
    model3 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=False)
    model3.preprocess_data(dataset_path, target_regression)
    model3.train_model()
    model3.evaluate_model()
    
    # Model 4: Regression with hyperparameter tuning
    print("\n" + "=" * 80)
    print("MODEL 4: REGRESSION WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    model4 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=False)
    model4.preprocess_data(dataset_path,target_regression)
    tuning_results = model4.tune_hyperparameters(type="grid_search",grid_size="small")
    model4.evaluate_model()

    # Model 5: Regression with hyperparameter tuning using random search
    print("\n" + "=" * 80)
    print("MODEL 5: REGRESSION WITH HYPERPARAMETER TUNING USING RANDOM SEARCH")
    print("=" * 80)
    model5 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=False)
    model5.preprocess_data(dataset_path,target_regression)
    tuning_results = model5.tune_hyperparameters(type="random_search")
    model5.evaluate_model()


    # Model 6: Classification with hyperparameter tuning using grid search
    print("\n" + "=" * 80)
    print("MODEL 6: CLASSIFICATION WITH HYPERPARAMETER TUNING USING GRID SEARCH")
    print("=" * 80)
    model6 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=True)
    model6.preprocess_data(dataset_path,target_classification)
    tuning_results = model6.tune_hyperparameters(type="grid_search",grid_size="small")
    model6.evaluate_model()

    # Model 7: Classification with hyperparameter tuning using random search
    print("\n" + "=" * 80)
    print("MODEL 7: CLASSIFICATION WITH HYPERPARAMETER TUNING USING RANDOM SEARCH")
    print("=" * 80)
    model7 = XGBoostModel(verbose=True, use_class_weights=False, is_classifier=True)
    model7.preprocess_data(dataset_path,target_classification)
    tuning_results = model7.tune_hyperparameters(type="random_search")
    model7.evaluate_model()

    # Model 8: Classification with hyperparameter tuning using random search with class weights
    print("\n" + "=" * 80)
    print("MODEL 8: CLASSIFICATION WITH HYPERPARAMETER TUNING USING RANDOM SEARCH WITH CLASS WEIGHTS")
    print("=" * 80)
    model8 = XGBoostModel(verbose=True, use_class_weights=True, is_classifier=True)
    model8.preprocess_data(dataset_path,target_classification)
    tuning_results = model8.tune_hyperparameters(type="random_search",use_class_weights=True)
    model8.evaluate_model()

    # Model 9: Regression with hyperparameter tuning using grid search with class weights
    print("\n" + "=" * 80)
    print("MODEL 9: REGRESSION WITH HYPERPARAMETER TUNING USING GRID SEARCH WITH CLASS WEIGHTS")
    print("=" * 80)
    model9 = XGBoostModel(verbose=True, use_class_weights=True, is_classifier=False)
    model9.preprocess_data(dataset_path,target_regression)
    tuning_results = model9.tune_hyperparameters(type="grid_search",grid_size="small",use_class_weights=True)
    model9.evaluate_model()

