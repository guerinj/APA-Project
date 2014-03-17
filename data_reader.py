import random
import string
import numpy as np


class DataReader(object):

    universe = list()

    def __init__(self, datasource, stopwords_source=None):
        words = set()
        with open(datasource, 'r') as data:
            for line in data:
                _, _, sentence = line.split('|')
                new_words = sentence.split(' ')
                for word in new_words:
                    if word.strip() not in string.punctuation:
                        words = words.union(set([word.strip().lower()]))
        if stopwords_source:
            stopwords = set()
            with open(stopwords_source, 'r') as data:
                for line in data:
                    stopwords = stopwords.union(set([line.strip().lower()]))
            words = words.difference(stopwords)
        self.universe = list(words)

    def get_sentence_coordinates(self, sentence):
        sentence_nparray = np.zeros((len(self.universe)+1))
        self.universe
        for word in sentence.split(' '):
            word = word.strip().lower()
            index = -1
            try:
                index = self.universe.index(word)
            except:
                pass
            if index != -1:
                sentence_nparray[index] = 1
        # Ajoute une coordonnes supplementaire pour eviter le produit scalaire nul
        sentence_nparray[len(self.universe)] = 1
        return sentence_nparray


if __name__ == '__main__':
    datareader = DataReader('data/test_data/test.data', 'data/stopwords/stopwords.txt')
    sentence = "7549 | 0.84722 | insightfully written , delicately performed"
    print datareader.get_sentence_coordinates(sentence)
