__author__ = 'barmalei4ik'

import threading
import csv
from collections import deque
import math
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
from neupy import algorithms, layers
from neupy.functions.errors import rmsle

class NeuralNetBacktest(threading.Thread):

    def __init__(self):

        np.random.seed(0)

        self.nnet = None
        self.data = []
        self.training_period = 100

        self.features = []
        self.targets = []

        self.features_stded = []
        self.targets_stded = []

        self.feature_data_scaler = None
        self.target_data_scaler = None

        self.ema12 = ExpMovingAverage(period=12)
        self.ema26 = ExpMovingAverage(period=26)
        self.macd9 = ExpMovingAverage(period=9)

        self._load_data()
        self._prep_indicators()
        self._standardize_features()
        self._initial_training()

    def _load_data(self):

        with open("data/EURCHF_day.csv", "rb") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "<TICKER>":
                    self.data.append([row[1], float(row[3]), float(row[4]), float(row[5]), float(row[6])])

        print "Loaded data: {0} rows".format(len(self.data))

    def _prep_indicators(self):

        for index, item in enumerate(self.data):
            close = item[4]

            ema12 = self.ema12(close) or close
            item.append(round(ema12, 4))

            ema26 = self.ema26(close) or close
            item.append(round(ema26, 4))

            macd9 = self.macd9(ema12 - ema26) or close
            item.append(round(macd9, 4))

            self.features.append([close, ema12, ema26, macd9])

            try:
                self.targets.append([self.data[index + 1][4]])
            except:
                pass

    def _standardize_features(self):
        self.feature_data_scaler = preprocessing.MinMaxScaler()
        self.target_data_scaler = preprocessing.MinMaxScaler()

        self.features_stded = self.feature_data_scaler.fit_transform(self.features)
        self.targets_stded = self.target_data_scaler.fit_transform(self.targets)

        back = self.feature_data_scaler.inverse_transform(self.features_stded)
        print "Done"


    def _initial_training(self):
        x_train, x_test, y_train, y_test = train_test_split(self.features_stded[:self.training_period],
                                                            self.targets_stded[:self.training_period],
                                                            train_size=0.85)



        cgnet = algorithms.ConjugateGradient(
            connection=[
                layers.SigmoidLayer(4),
                layers.SigmoidLayer(50),
                layers.OutputLayer(1),
            ],
            search_method='golden',
            show_epoch=100,
            verbose=True,
            optimizations=[algorithms.LinearSearch],
        )

        cgnet.train(x_train, y_train, x_test, y_test, epochs=300)

        y_predict = cgnet.predict(x_test).round(5)

        print len(y_test), len(y_predict)

        test_normalized = self.feature_data_scaler.inverse_transform(y_test)
        predicted_normalized = self.target_data_scaler.inverse_transform(y_predict)

        error = rmsle(self.feature_data_scaler.inverse_transform(y_test),
                      self.target_data_scaler.inverse_transform(y_predict))

        print "Error on initial training: {0}".format(error)


    def _subsequent_training(self):
        pass

    def create_network(self):
        pass

    def run(self):
        pass


class ExpMovingAverage():

    def __init__(self, period):

        self.period = period
        self.stream = deque()
        self.multiplier = 2.0 / float((period + 1))
        self.prev_ema = None
        self.ema = None

    #end


    def __call__(self, value):

        self.stream.append(value)

        if len(self.stream) > self.period:
            self.stream.popleft()

            if not self.prev_ema:

                self.prev_ema = sum(self.stream) / len(self.stream)

            else:

                self.ema = (value - self.prev_ema) * self.multiplier + self.prev_ema
                self.prev_ema = self.ema

        return self.prev_ema

    #end


if __name__ == "__main__":

    NeuralNetBacktest().start()