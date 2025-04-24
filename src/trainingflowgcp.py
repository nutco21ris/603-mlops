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
class GoldDemandTrainFlow(FlowSpec):
    """
    A flow for training a gold demand region classifier model.
    The flow includes data loading, preprocessing, model training, and MLflow tracking.
    """
    
    # Define parameters
    random_state = Parameter('random_state', 
                           help='Random seed for reproducibility',
                           default=42)
    
    test_size = Parameter('test_size',
                         help='Proportion of dataset to include in the test split',
                         default=0.3)
    
    @kubernetes(cpu=0.1, memory=2048)
    @timeout(minutes=10)
    @retry(times=3)
    @step
    def start(self):
        """
        Create and preprocess the gold demand dataset
        """
        # Create sample data
        countries = [
            'China', 'India', 'US', 'Germany', 'Turkey',
            'Japan', 'UK', 'France', 'UAE', 'Thailand',
            'Switzerland', 'Hong Kong', 'Singapore', 'Taiwan', 'South Korea',
            'Malaysia', 'Indonesia', 'Vietnam', 'Saudi Arabia', 'Egypt',
            'Brazil', 'Mexico', 'Canada', 'Spain', 'Italy',
            'Netherlands', 'Belgium', 'Austria', 'Poland', 'Finland'
        ]
        
        # Generate quarterly data (16 quarters: Q1'20 to Q4'23)
        quarters = [f"Q{q}'2{y}" for y in range(0,4) for q in range(1,5)]
        
        # Create base demand values for each region
        np.random.seed(self.random_state)
        base_demands = {
            'Asia': 120,
            'Europe': 100,
            'Americas': 90,
            'Middle East': 110,
            'Other': 80
        }
        
        # Generate data with regional patterns
        quarterly_data = []
        for country in countries:
            # Determine region
            if country in ['China', 'India', 'Japan', 'Thailand', 'Vietnam', 'South Korea', 'Indonesia', 
                         'Malaysia', 'Singapore', 'Taiwan', 'Hong Kong']:
                base = base_demands['Asia']
            elif country in ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Spain', 'Poland', 
                           'Netherlands', 'Belgium', 'Austria', 'Finland']:
                base = base_demands['Europe']
            elif country in ['US', 'Canada', 'Brazil', 'Mexico']:
                base = base_demands['Americas']
            elif country in ['Turkey', 'UAE', 'Saudi Arabia', 'Egypt']:
                base = base_demands['Middle East']
            else:
                base = base_demands['Other']
            
            # Generate country data with seasonal patterns and trends
            country_data = []
            for i, q in enumerate(quarters):
                seasonal = 10 * np.sin(2 * np.pi * (i % 4) / 4)  # seasonal pattern
                trend = i * 0.5  # slight upward trend
                noise = np.random.normal(0, 5)  # random variation
                value = base + seasonal + trend + noise
                country_data.append(value)
            quarterly_data.append(country_data)
        
        # Create DataFrame
        data = pd.DataFrame(quarterly_data, columns=quarters)
        data['Countries'] = countries
        
        print(f"Created dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Define region mapping function
        def map_to_region(country):
            asia = ['China', 'India', 'Japan', 'Thailand', 'Vietnam', 'South Korea', 'Indonesia', 
                    'Malaysia', 'Singapore', 'Philippines', 'Taiwan', 'Hong Kong']
            europe = ['Germany', 'France', 'UK', 'Italy', 'Switzerland', 'Spain', 'Poland', 
                      'Netherlands', 'Belgium', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Austria']
            americas = ['US', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru']
            middle_east = ['Turkey', 'UAE', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Oman', 'Bahrain', 'Egypt']
            
            if country in asia:
                return 'Asia'
            elif country in europe:
                return 'Europe'
            elif country in americas:
                return 'Americas'
            elif country in middle_east:
                return 'Middle East'
            else:
                return 'Other'

        # Apply mapping and create target variable
        data['Region'] = data['Countries'].apply(map_to_region)
        print(f"Region distribution:\n{data['Region'].value_counts()}")

        # Prepare features and target
        X = data.iloc[:, :-2]  # All quarters data excluding Countries and Region
        y = data['Region']      # Target is region

        # Print feature names for reference
        self.feature_names = X.columns.tolist()
        print(f"Features used: {self.feature_names}")

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        
        self.next(self.train_model)

    @kubernetes(cpu=0.1, memory=2048)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def train_model(self):
        """
        Train a random forest model
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        import mlflow
        import os

        # Create and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=20, 
                max_depth=3,
                random_state=self.random_state
            ))
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        train_score = pipeline.score(self.X_train, self.y_train)
        test_score = pipeline.score(self.X_test, self.y_test)
        
        # Log to MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment("gold_demand_classification_gcp")
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("n_estimators", 20)
            mlflow.log_param("max_depth", 3)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            
            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name="gold-demand-classifier-gcp"
            )
            
            self.run_id = run.info.run_id
            
        print(f"\nTraining Results:")
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"MLflow run ID: {self.run_id}")
        
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
    GoldDemandTrainFlow() 