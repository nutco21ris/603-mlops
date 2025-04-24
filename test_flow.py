from metaflow import FlowSpec, step, kubernetes

class TestFlow(FlowSpec):
    
    @kubernetes
    @step
    def start(self):
        print("Starting the flow")
        self.next(self.end)
    
    @kubernetes
    @step
    def end(self):
        print("Flow completed")

if __name__ == '__main__':
    TestFlow() 