#! /usr/bin/env python3
import argparse
import pandas as pd

from imbalancedtensorflow.imbalanced_tensorflow import ImbalancedTensorflow
from imbalancedtensorflow.data_generator import DataGenerator


class MakePredictions:
    def __init__(self,
                 parent_folder=None,
                 output_file="tensorflowresults.csv"):
        self.results = None
        self.output_file = output_file
        self.imbalanced = ImbalancedTensorflow()
        self.parent_folder = parent_folder

    def generate_predictions(self):
        print("generate predictions")

        generator = DataGenerator()
        generator.parent_folder = self.parent_folder
        generator.generator()
        test = generator.test

        n = self.imbalanced.neuralnetwork(df=test)

        for x in [0, 1, 2, 3, 4]:
            self.imbalanced.plotter(n[x]['model'], n[x]['testfeatures'], n[x]['test_labels'], n[x]['test_predictions'])

        main_dataframe = []
        for x in [0, 1, 2, 3, 4]:
            results = n[5][x]
            results['True'] = n[x]['test_labels']
            results['Predicted'] = n[x]['test_predictions']
            results = pd.merge(n[6][['Country_x', 'Country_y', 'GDP per capita_x', 'GDP per capita_y']], results,
                               on=['GDP per capita_x', 'GDP per capita_y'])  # [['Country_x','Country_y']]
            main_dataframe.append(results)
        self.results = pd.concat(main_dataframe)
        print('Results generated')

    def save_csv(self):
        self.results.to_csv('results/%s' % self.output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_folder', help="parent folder with data", default='data')
    parser.add_argument('--output_file', help="File to save results to", default='tensorflowresults.csv')
    args = parser.parse_args()

    predictor = MakePredictions(parent_folder=args.parent_folder,
                                output_file=args.output_file)
    print('generate predictions')
    predictor.generate_predictions()
    predictor.save_csv()


if __name__ == "__main__":
    main()
