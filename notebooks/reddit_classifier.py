from metaflow import FlowSpec, step

class RedditClassifier(FlowSpec):
    def start(self):
        print("Flow is starting")
        self.next(self.load_data)

    def load_data(self):
        import pandas as pd
        import numpy

        print("Data is loading")
        self.x_sample = pd.read_csv('sample_reddit.csv', header=None).to_numpy().reshape((-1,))
        print("Data is loaded")
        self.next(self.load_model)

    def load_model(self):
        import joblib
        print("Pipeline loading")
        self.loaded_pipeline = joblib.load("reddit_model_pipeline.joblib")
        print("Pipeline loaded")
        self.next(self.predict_class)

    def predict_class(self):
        print("Making predictions")
        self.predictions = self.loaded_pipeline.predict_proba(self.x_sample)
        print("Predictions made")
        self.next(self.save_results)

    def save_results(self):
        import pandas as pd
        print("Saving results")
        pd.DataFrame(self.predictions).to_csv("sample_preds.csv", index=None, header=None)
        print("Results saved")
        self.next(self.end)
    
    def end(self):
        print("Flow is ending")

if __name__ == '__main__':
    RedditClassifier() 