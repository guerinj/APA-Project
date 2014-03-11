import numpy as np
import random
import math
import time

from perceptron import Perceptron
from data_reader import DataReader


class APAProject(object):

    def __init__(self):
        self.data_reader = DataReader('data/training_data/training.data')
        self.perceptron = Perceptron()
        # Let's create 5 classifiers
        universe_size = len(self.data_reader.universe)
        self.classifiers = [np.zeros((universe_size)) for i in range(5)]

    def file_to_data_set(self, file):
        data_set = []
        with open(file) as data:
            for line in data:
                _, score, sentence = line.split('|')
                score = float(score)

                # Calculating train target:
                # 0 if 0 < score <= 0.2, 1 if 0.2 < score <= 0.4, etc...
                class_number = math.floor(score * 5)
                sentence_vector = self.data_reader.get_sentence_coordinates(sentence)
                data_set.append((sentence_vector, class_number))
        return data_set

    def train(self):
        start_time = time.time()

        print "Starting training session ..."

        # We need to read data from datasmall and train the perceptron
        training_data_set = self.file_to_data_set('data/training_data/training.data')

        PERIODS = 3

        for i in range(PERIODS):
            # For each period, reshuffle
            random.shuffle(training_data_set)
            # We train every classfier
            for (classifier_index, classifier) in enumerate(self.classifiers):
                self.classifiers[classifier_index], updates = self.perceptron.train_epoch(training_data_set, classifier_index, classifier)

        training_end_time = time.time()
        training_duration = training_end_time - start_time
        print "Training session finished: duration %s seconds" % training_duration

    def test(self):
        print "Starting testing session..."

        test_data_set = self.file_to_data_set('data/test_data/test.data')

        for (classifier_index, classifier) in enumerate(self.classifiers):
            error_count, success_count = self.perceptron.test_classifier(test_data_set, classifier, classifier_index)
            print "Classifier %s just finished. %s%% results are good" % ((classifier_index + 1), success_count * 100 / (success_count + error_count))


if __name__ == '__main__':
    apa_project = APAProject()
    apa_project.train()
    apa_project.test()
