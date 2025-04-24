from metaflow import FlowSpec, step, kubernetes, timeout, retry, catch

class ClassifierTrainFlow(FlowSpec):

    @kubernetes(cpu=0.1, memory=512)
    @timeout(minutes=10)
    @retry(times=3)
    @step
    def start(self):
        import subprocess
        subprocess.check_call(['pip', 'install', 'numpy==1.26.4', 'scikit-learn==1.5.1'])

        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        X = X[:20]
        y = y[:20]
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.lambdas = np.array([0.001, 0.1])
        self.next(self.train_lasso, foreach='lambdas')

    @kubernetes(cpu=0.1, memory=512)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def train_lasso(self):
        import subprocess
        subprocess.check_call(['pip', 'install', 'numpy==1.26.4', 'scikit-learn==1.5.1'])

        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @kubernetes(cpu=0.1, memory=512)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def choose_model(self, inputs):
        import subprocess
        subprocess.check_call(['pip', 'install', 'numpy==1.26.4', 'scikit-learn==1.5.1'])

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @kubernetes(cpu=0.1, memory=512)
    @timeout(minutes=10)
    @retry(times=3)
    @catch(var='error')
    @step
    def end(self):
        import subprocess
        subprocess.check_call(['pip', 'install', 'numpy==1.26.4', 'scikit-learn==1.5.1'])

        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    ClassifierTrainFlow() 