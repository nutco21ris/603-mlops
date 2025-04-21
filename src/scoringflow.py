from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
import mlflow
import os

class WineScoringFlow(FlowSpec):
    """
    A flow for scoring new wine samples using the trained classifier model.
    """
    
    # Define parameters
    data_path = Parameter('data_path',
                         help='Path to the CSV file containing new wine samples',
                         required=True)
    
    model_stage = Parameter('model_stage',
                          help='Stage of the model to use (None, Staging, Production)',
                          default='Production')
    
    @step
    def start(self):
        """
        Load the new data and the trained model
        """
        # Load new data
        try:
            self.data = pd.read_csv(self.data_path, sep=';')
            print(f"Loaded {len(self.data)} samples for prediction")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
        # Set up MLflow to use local directory
        os.environ['MLFLOW_TRACKING_URI'] = 'mlruns'
        
        # Load the model from the local MLflow store
        try:
            self.model = mlflow.sklearn.load_model(f"models:/wine-quality-classifier/{self.model_stage}")
            print(f"Loaded model from {self.model_stage} stage")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying to load latest model version...")
            latest_run = mlflow.search_runs(experiment_names=["wine-quality-metaflow"]).iloc[0]
            model_path = latest_run.artifact_uri + "/model"
            self.model = mlflow.sklearn.load_model(model_path)
            print("Loaded latest model version")
            
        self.next(self.make_predictions)
        
    @step
    def make_predictions(self):
        """
        Make predictions on the new data
        """
        # Make predictions
        predictions = self.model.predict(self.data)
        probabilities = self.model.predict_proba(self.data)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'prediction': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1]
        })
        
        print("Made predictions successfully")
        self.next(self.end)
        
    @step
    def end(self):
        """
        Save predictions and end the flow
        """
        # Save predictions to CSV
        output_path = 'predictions.csv'
        self.results.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total samples: {len(self.results)}")
        print(f"Predicted good quality wines: {sum(self.results['prediction'] == 1)}")
        print(f"Predicted regular quality wines: {sum(self.results['prediction'] == 0)}")

if __name__ == '__main__':
    WineScoringFlow() 