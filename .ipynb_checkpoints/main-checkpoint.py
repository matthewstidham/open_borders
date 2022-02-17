#! /usr/bin/env python3
import argparse

from imbalancedtensorflow.imbalanced_tensorflow import ImbalancedTensorflow

class MakePredictions(file=None,
                     output_file="tensorflowresults.csv"):
    def __init__(self):
        self.file = file
        self.results = None
        self.output_file = output_file
    
    def generate_predictions(self):
        n=neuralnetwork(test)

        for x in [0,1,2,3,4]:
            plotter(n[x]['model'],n[x]['testfeatures'],n[x]['test_labels'],n[x]['test_predictions'])

        l=[]
        for x in [0,1,2,3,4]:
            results=n[5][x]
            results['True']=n[x]['test_labels']
            results['Predicted']=n[x]['test_predictions']
            results=pd.merge(n[6][['Country_x','Country_y','GDP per capita_x','GDP per capita_y']],results,on=['GDP per capita_x','GDP per capita_y'])  #[['Country_x','Country_y']]
            l.append(results)
        self.results=pd.concat(l)
        print('Results generated')
    
    def save_csv(self):
        self.results.to_csv('results/%s' % self.output_file)
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', description="File to parse",required=True)
    parser.add_argument('--output_file', description="File to save results to", default='tensorflowresults.csv')
    args = parser.parse_args()
    
    predictor = MakePredictions(file=args.file,
                               output_file=args.output_file)
    predictor.generate_predictions()
    predictor.save_csv()
    