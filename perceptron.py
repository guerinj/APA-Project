import numpy as np


class Perceptron(object):

    def train_step(self, observation, classifier, perceptron_class):
        y = np.dot(observation, classifier)

        if y * perceptron_class > 0:
            return False, classifier
        else:
            return True, np.add(classifier, observation * perceptron_class)

    def train_epoch(self, shuffled_set, classifier_class, classifier):
        train_classifier = classifier
        updates = 0

        for (vector, vector_class) in shuffled_set:
            changed, train_classifier = self.train_step(
                vector,
                train_classifier,
                1 if classifier_class == vector_class else -1
            )
            if changed:
                updates += 1

        return train_classifier, updates

    def test_classifier(self, test_set, classifier, classifier_class):
        error = 0
        success = 0
        for (overvation_vector, observation_class) in test_set:
            result = np.dot(classifier, overvation_vector)
            observation_belongs_to_classifier_class = 1 if observation_class == classifier_class else -1

            if result * observation_belongs_to_classifier_class > 0:
                success += 1
            else:
                error += 1
        return error, success
