from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

class WineTrainingFlow(FlowSpec):
    """
    A flow for training a wine quality classifier model.
    The flow includes data loading, preprocessing, model training, and MLflow tracking.
    """
    
    # Define parameters
    random_state = Parameter('random_state', 
                           help='Random seed for reproducibility',
                           default=42)
    
    test_size = Parameter('test_size',
                         help='Proportion of dataset to include in the test split',
                         default=0.2)
    
    n_estimators = Parameter('n_estimators',
                           help='Number of trees in random forest',
                           default=100)
    
    @step
    def start(self):
        """
        Load and preprocess the wine quality dataset
        """
        # Load data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        
        # Create binary classification target (good quality >= 6)
        df['quality'] = (df['quality'] >= 6).astype(int)
        
        # Split features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Save scaler for later use
        self.scaler = scaler
        
        print("Data loaded and preprocessed successfully")
        self.next(self.train_model)
        
    @step
    def train_model(self):
        """
        Train the random forest model and track with MLflow
        """
        # Set up MLflow to use local directory
        os.environ['MLFLOW_TRACKING_URI'] = 'mlruns'
        mlflow.set_experiment('wine-quality-metaflow')
        
        # Train model
        with mlflow.start_run() as run:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Get predictions and metrics
            train_score = self.model.score(self.X_train_scaled, self.y_train)
            test_score = self.model.score(self.X_test_scaled, self.y_test)
            
            # Log parameters
            mlflow.log_param('random_state', self.random_state)
            mlflow.log_param('n_estimators', self.n_estimators)
            mlflow.log_param('test_size', self.test_size)
            
            # Log metrics
            mlflow.log_metric('train_accuracy', train_score)
            mlflow.log_metric('test_accuracy', test_score)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                'model',
                registered_model_name='wine-quality-classifier'
            )
            
            self.run_id = run.info.run_id
            
        print(f"Model trained successfully. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        self.next(self.end)
        
    @step
    def end(self):
        """
        End the flow
        """
        print("Flow completed successfully!")
        print(f"MLflow run ID: {self.run_id}")

if __name__ == '__main__':
    WineTrainingFlow() 