# Sujet TP1 : http://perso.limsi.fr/allauzen/APA_tp_perceptron.pdf
# Sujet TP2 : http://perso.limsi.fr/allauzen/APA_tp_multiclasse.pdf

import cPickle
import gzip
import numpy
import random
import numpy as np
import matplotlib.pyplot as plt

import data_reader


# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

datareader = DataReader()

#
#  1. Analyse du corpus
#

print "1.2 La dimension des images est 28x28"

print "1.3 Nombre d'images disponibles, "
print "\t pour les tests : %s" % test_set[0].shape[0]
print "\t pour l'apprentissage : %s" % train_set[0].shape[0]

print "1.4 Les valeurs des pixels vont de 0 a 1."


#
#  2. Perceptron : premiers pas
#

train_size = 1000
trained_target = 7
train_order = range(train_size)
random.shuffle(train_order)
classifier = np.zeros((784))

print "2.5 Ordre d'apprentissage : "
print train_order[:10]

print "2.6 Fonction pas d'apprentissage"


def train_step(image, classifier, image_class):
    y = np.dot(image, classifier)

    if y * image_class > 0:
        return False, classifier
    else:
        return True, np.add(classifier, image * image_class)


print "2.7 Fonction epoque d'apprentissage"


def train_epoch(train_set, trained_target, order, classifier):
    train_classifier = classifier
    updates = 0

    for index in order:
        changed, train_classifier = train_step(
            train_set[0][index],
            train_classifier,
            1 if trained_target == train_set[1][index] else -1
        )

        if changed:
            updates += 1

    return train_classifier, updates

print "2.8 Fonction de test"


def test_classifier(test_set, classifier, target):
    error = 0
    success = 0
    for index, image in enumerate(test_set[0]):
        result = np.dot(classifier, image)
        label = 1 if test_set[1][index] == target else -1

        if result * label > 0:
            success += 1
        else:
            error += 1

    # print "Test effectues : %s/%s erreurs soit %s pourcent" % (error, success+error, error*100.0/(success+error) )
    return error, success

print "Test des fonctions precedentes :"

small_train_set = [
    train_set[0][:train_size],
    train_set[1][:train_size]
]

small_test_set = [
    test_set[0][:train_size],
    test_set[1][:train_size]
]

classifiers = []

results = [[0] * 10, [0] * 10, [0] * 10]
for target in range(10):
    result = []
    print "\n\n\n"
    print "Apprentissage pour %s" % target
    print "1er passage"
    classifier, updates = train_epoch(small_train_set, target, train_order, classifier)
    e, s = test_classifier(small_train_set, classifier, target)

    print "2eme passage"
    random.shuffle(train_order)
    classifier, updates = train_epoch(small_train_set, target, train_order, classifier)
    e, s = test_classifier(small_train_set, classifier, target)

    print "3eme passage"
    random.shuffle(train_order)
    classifier, updates = train_epoch(small_train_set, target, train_order, classifier)
    e, s = test_classifier(small_train_set, classifier, target)

    classifiers.append(classifier)

#
#  3. Courbe d'apprentissage
#

print "3.9 Courbe d'ordre d'apprentissage"

for train_size, color in [(100, 'r'), (1000, 'b'), (5000, 'g'), (10000, 'c')]:
    print train_size
    values_x = []
    values_y = []
    values_error = []
    trained_target = 6

    random.shuffle(train_order)
    classifier = np.zeros((784))

    plot_train_set = [
        train_set[0][:train_size],
        train_set[1][:train_size]
    ]

    plot_test_set = [
        test_set[0][:train_size],
        test_set[1][:train_size]
    ]

    for e in range(1, 25):
        train_order = range(train_size)
        classifier, updates = train_epoch(
            plot_train_set,
            trained_target,
            train_order,
            classifier)

        values_x.append(e)
        values_y.append(updates)

        errors, success = test_classifier(
            plot_test_set,
            classifier,
            trained_target
        )
        values_error.append(errors)

        plt.subplot(211)
        plt.plot(values_x, values_y, color + '-')
        plt.subplot(212)
        plt.plot(values_x, values_error, color + '--')
    print values_error


plt.show()

#
#   TP2 : un contre tous
#

results = []
for i in range(10):
    results.append(np.dot(classifiers[i], test_set[0][2]))
print "On obtient :"
print results


# Afficher une image
im = test_set[0][2].reshape(28, 28)
plt.imshow(im.reshape(28, 28), plt.cm.gray)
plt.show()
