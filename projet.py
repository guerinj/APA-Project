import numpy as np
import random
import math
import time

from perceptron import Perceptron
from softmax import Softmax
from data_reader import DataReader


class APAProject(object):

    def __init__(self):
        self.data_reader = DataReader('data/training_data/training.data', 'data/stopwords/stopwords.txt')
        self.perceptron = Perceptron()
        self.softmax = Softmax()
        # Let's create 5 classifiers
        universe_size = len(self.data_reader.universe)
        self.perceptron_classifiers = [np.zeros((universe_size + 1)) for i in range(5)]
        self.softmax_classifier = np.ones((5, universe_size + 1))

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

    def train_perceptron(self):
        start_time = time.time()

        print "Starting training session ..."

        # We need to read data from datasmall and train the perceptron
        training_data_set = self.file_to_data_set('data/training_data/training.data')

        PERIODS = 3

        for i in range(PERIODS):
            # For each period, reshuffle
            random.shuffle(training_data_set)
            # We train every classfier
            for (classifier_index, classifier) in enumerate(self.perceptron_classifiers):
                self.perceptron_classifiers[classifier_index], updates = self.perceptron.train_epoch(training_data_set, classifier_index, classifier)

        training_end_time = time.time()
        training_duration = training_end_time - start_time
        print "Training session finished: duration %s seconds" % training_duration

    def test_perceptron(self):
        print "Starting testing session..."

        test_data_set = self.file_to_data_set('data/test_data/test.data')

        for (classifier_index, classifier) in enumerate(self.perceptron_classifiers):
            error_count, success_count = self.perceptron.test_classifier(test_data_set, classifier, classifier_index)
            print "Classifier %s just finished. %s%% results are good" % ((classifier_index + 1), success_count * 100 / (success_count + error_count))

    def test_perceptron_multiclass(self):
        print "Starting testing session..."

        test_data_set = self.file_to_data_set('data/test_data/test.data')

        success_count = 0
        error_count = 0

        for (sentence_vector, class_number) in test_data_set:
            results_classifiers = []
            test_class = -1
            for (classifier_index, classifier) in enumerate(self.perceptron_classifiers):
                results_classifiers.append(np.dot(classifier, sentence_vector))
            if results_classifiers.index(max(results_classifiers)) == class_number:
                success_count += 1
            else:
                error_count += 1

        print "Classifier just finished. %s/%s ~= %s%% results are good" % (success_count, (error_count + success_count), success_count * 100 / (success_count + error_count))

    def train_softmax(self):
        start_time = time.time()
        print "Starting softmax training session..."

        # We need to read data from datasmall and train the perceptron
        training_data_set = self.file_to_data_set('data/training_data/training.data')

        PERIODS = 10

        for i in range(PERIODS):
            random.shuffle(training_data_set)
            # On apprend PERIODS fois et a chaque passage on test le classifier pour etudier l'evolution
            # Rappel : self.softmax_classifier = np.ones((5, universe_size))
            self.softmax_classifier = self.softmax.train_epoch(self.softmax_classifier, training_data_set)
            self.test_softmax()

        training_end_time = time.time()
        training_duration = training_end_time - start_time
        print "Training session finished: duration %s seconds" % training_duration

    def test_softmax(self):
        print "Starting softmax testing session..."

        test_data_set = self.file_to_data_set('data/test_data/test.data')

        error_count, success_count = self.softmax.test_classifier(self.softmax_classifier, test_data_set)
        print "Classifier just finished. %s/%s ~= %s%% results are good" % (success_count, (error_count + success_count), success_count * 100 / (success_count + error_count))


if __name__ == '__main__':
    apa_project = APAProject()
    apa_project.train_perceptron()
    apa_project.test_perceptron()
    # apa_project.test_perceptron_multiclass()

    # apa_project.train_softmax() # apa_project.test_softmax() est appele apres chaque training pour evaluer les ameliorations
    # apa_project.test_softmax()
