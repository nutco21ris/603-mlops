from metaflow import FlowSpec, step, kubernetes, timeout, retry, catch, conda_base, Parameter
import pandas as pd
import numpy as np
import mlflow
import os

@conda_base(libraries={
    'numpy':'1.26.4', 
    'scikit-learn':'1.5.1', 
    'pandas':'2.2.2',
    'mlflow':'2.15.1'
}, python='3.9.16')
class GoldDemandScoringFlow(FlowSpec):
    """
    A flow for scoring new data using the trained gold demand classifier model.
    """
    
    # Define parameters
    model_stage = Parameter('model_stage',
                          help='Stage of the model to use (None, Staging, Production)',
                          default='None')
    
    @kubernetes(cpu=0.1, memory=2048)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def start(self):
        """
        Create sample data for scoring
        """
        # Create sample data
        countries = [
            'Vietnam', 'Singapore', 'Spain', 'UAE',  # One from each region
            'Australia'  # Other
        ]
        
        # Generate quarterly data (16 quarters: Q1'20 to Q4'23)
        quarters = [f"Q{q}'2{y}" for y in range(0,4) for q in range(1,5)]
        
        # Generate random data
        np.random.seed(42)
        quarterly_data = []
        for _ in countries:
            country_data = []
            for _ in quarters:
                value = np.random.normal(100, 20)
                country_data.append(value)
            quarterly_data.append(country_data)
        
        # Create DataFrame
        self.data = pd.DataFrame(quarterly_data, columns=quarters)
        self.data['Countries'] = countries
        
        print(f"Created scoring dataset with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        
        self.next(self.score_data)

    @kubernetes(cpu=0.1, memory=2048)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def score_data(self):
        """
        Score the data using the trained model
        """
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        
        # Load the model
        model = mlflow.sklearn.load_model(
            f"models:/gold-demand-classifier-gcp/{self.model_stage}"
        )
        
        # Prepare features (exclude Countries column)
        X = self.data.iloc[:, :-1]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'Country': self.data['Countries'],
            'Predicted_Region': predictions
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(model.classes_):
            self.results[f'Probability_{class_name}'] = probabilities[:, i]
        
        print("\nPrediction Results:")
        print(self.results)
        
        self.next(self.end)

    @kubernetes(cpu=0.1, memory=2048)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def end(self):
        """
        End the flow
        """
        print("Flow completed successfully!")

if __name__ == '__main__':
    GoldDemandScoringFlow() 