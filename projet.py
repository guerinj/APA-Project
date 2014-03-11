import numpy as np
import random
import math
import time

from perceptron import Perceptron
from data_reader import DataReader

start_time = time.time()

print "Starting training session ..."

data_reader = DataReader('data/training_data/training.data')
perceptron = Perceptron()

universe_size = len(data_reader.universe)

# Let's create 5 classifiers
classifiers = [np.zeros((universe_size)) for i in range(5)]


# We need to read data from datasmall and train the perceptron

training_data_set = []

with open('data/training_data/training.data') as data:
    for line in data:
        _, score, sentence = line.split('|')
        score = float(score)

        # Calculating train target:
        # 0 if 0 < score <= 0.2, 1 if 0.2 < score <= 0.4, etc...
        class_number = math.floor(score * 5)
        sentence_vector = data_reader.get_sentence_coordinates(sentence)
        training_data_set.append((sentence_vector, class_number))

# Now data set is filled with (vector, score)

# Shuffling training_data_set

PERIODS = 3

for i in range(PERIODS):
    # For each period, reshuffle
    random.shuffle(training_data_set)
    # We train every classfier
    for (classifier_index, classifier) in enumerate(classifiers):
        classifiers[classifier_index], updates = perceptron.train_epoch(training_data_set, classifier_index, classifier)

training_end_time = time.time()
training_duration = training_end_time - start_time
print "Training session finished: duration %s seconds" % training_duration
print "Starting testing session..."

test_data_set = []

with open('data/test_data/test.data') as data:
    for line in data:
        _, score, sentence = line.split('|')
        score = float(score)

        class_number = math.floor(score * 5)
        sentence_vector = data_reader.get_sentence_coordinates(sentence)
        test_data_set.append((sentence_vector, class_number))

for (classifier_index, classifier) in enumerate(classifiers):
    error_count, success_count = perceptron.test_classifier(test_data_set, classifier, classifier_index)
    print "Classifier %s just finished. %s%% results are good" % ((classifier_index + 1), success_count * 100 / (success_count + error_count))
