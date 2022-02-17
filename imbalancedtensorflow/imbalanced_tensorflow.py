#! /usr/bin/env python3

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class ImbalancedTensorflow:
    def __init__(self):
        self.baseline_history = None
        self.EPOCHS = 100
        self.BATCH_SIZE = 2048
        self.METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

    @staticmethod
    def plot_loss(history, label, n):
        # Use a log scale to show the wide range of values.
        plt.semilogy(history.epoch, history.history['loss'],
                     color=colors[n], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_loss'],
                     color=colors[n], label='Val ' + label,
                     linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend()

    @staticmethod
    def plot_metrics(history):
        metrics = ['loss', 'auc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()

    def make_model(self, output_bias=None, train_features=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = keras.Sequential([
            keras.layers.Dense(
                16, activation='relu',
                input_shape=(train_features.shape[-1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid',
                               bias_initializer=output_bias),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=self.METRICS)

        return model

    def tensor(self, train_df, test_df, neg, pos):
        train_df, val_df = train_test_split(train_df, test_size=.2)
        # Form np arrays of labels and features.
        train_labels = np.array(train_df['Status'])
        train_df = train_df.drop('Status', 1)
        val_labels = np.array(val_df['Status'])
        val_df = val_df.drop('Status', 1)
        test_labels = np.array(test_df['Status'])
        test_df = test_df.drop('Status', 1)

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        test_features = np.array(test_df)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)
        test_features = np.clip(test_features, -5, 5)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)
        model = self.make_model(train_features=train_features)
        model.evaluate(train_features, train_labels, batch_size=self.BATCH_SIZE, verbose=0)
        initial_bias = np.log([pos / neg])
        model = self.make_model(output_bias=initial_bias, train_features=train_features)
        model.evaluate(train_features, train_labels, batch_size=self.BATCH_SIZE, verbose=0)
        initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
        model.save_weights(initial_weights)
        model = self.make_model(train_features=train_features)
        model.load_weights(initial_weights)
        model.layers[-1].bias.assign([0.0])
        model.fit(  # zero_bias_history
            train_features,
            train_labels,
            batch_size=self.BATCH_SIZE,
            epochs=20,
            validation_data=(val_features, val_labels),
            verbose=0)
        model = self.make_model(train_features=train_features)
        model.load_weights(initial_weights)
        model.fit(  # careful_bias_history
            train_features,
            train_labels,
            batch_size=self.BATCH_SIZE,
            epochs=20,
            validation_data=(val_features, val_labels),
            verbose=0)
        model = self.make_model(train_features=train_features)
        model.load_weights(initial_weights)
        model.fit(  # baseline_history
            train_features,
            train_labels,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[early_stopping],
            validation_data=(val_features, val_labels))
        model.predict(train_features, batch_size=self.BATCH_SIZE)  # train_predictions_baseline
        test_predictions_baseline = model.predict(test_features, batch_size=self.BATCH_SIZE)
        d = dict()
        d['testfeatures'] = test_features
        d['test_labels'] = test_labels
        d['train_labels'] = train_labels
        d['test_predictions'] = test_predictions_baseline
        d['model'] = model
        return d

    def neuralnetwork(self, df=None):
        eps = 0.001
        for x in ['GDP_x', 'GDP_y', 'Population_x', 'Population_y', 'GDP', 'Population', 'Average Population',
                  'GDP per capita', 'worse GDP per capita', 'better GDP per capita', 'GDP per capita_x',
                  'GDP per capita_y']:
            df[x] = np.log(df.pop(x) + eps)
        raw_df = df._get_numeric_data()
        neg, pos = np.bincount(raw_df['Status'])
        cleaned_df = raw_df.copy()
        # Use a utility from sklearn to split and shuffle our dataset.
        shuffled = raw_df.sample(frac=1)
        result = np.array_split(shuffled, 5)
        run1 = self.tensor(pd.concat([result[x] for x in [1, 2, 3, 4]]), result[0], neg, pos)
        run2 = self.tensor(pd.concat([result[x] for x in [0, 2, 3, 4]]), result[1], neg, pos)
        run3 = self.tensor(pd.concat([result[x] for x in [0, 1, 3, 4]]), result[2], neg, pos)
        run4 = self.tensor(pd.concat([result[x] for x in [0, 1, 2, 4]]), result[3], neg, pos)
        run5 = self.tensor(pd.concat([result[x] for x in [0, 1, 2, 3]]), result[4], neg, pos)
        return [run1, run2, run3, run4, run5, result, df]

    @staticmethod
    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        print('True Negatives: ', cm[0][0])
        print('False Positives: ', cm[0][1])
        print('False Negatives: ', cm[1][0])
        print('True Positives: ', cm[1][1])
        print('Total Positive cases: ', np.sum(cm[1]))

    def plotter(self, model, test_features, test_labels, test_predictions_baseline):
        baseline_results = model.evaluate(test_features, test_labels,
                                          batch_size=self.BATCH_SIZE, verbose=0)
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value)
        print()

        self.plot_cm(test_labels, test_predictions_baseline)
