import numpy as np
import random

from perceptron import Perceptron
from data_reader import DataReader

data_reader = DataReader()
perceptron = Perceptron()

universe_size = len(data_reader.universe)

# Let's create 5 classifiers
classifiers = [np.zeros((universe_size)) for i in range(5)]


# We need to read data from datasmall and train the perceptron

data_set = []

with open('data/datasmall') as data:
    for line in data:
        _, score, sentence = line.split('|')
        score = float(score)

        # Calculating train target:
        # 0 if 0 < score <= 0.2, 1 if 0.2 < score <= 0.4, etc...
        class_number = floor(score * 5)
        sentence_vector = data_reader.get_sentence_coordinates(sentence)
        data_set.append((sentence_vector, class_number))

# Now data set is filled with (vector, score)

# Shuffling data_set

PERIODS = 3

for i in range(PERIODS):
    # For each period, reshuffle
    random.shuffle(data_set)
    # We train every classfier
    for (classifier_index, classifier) in enumerate(classifiers):
        classifiers[classifier_index], updates = perceptron.train_epoch(data_set, classifier_index, classifier)
