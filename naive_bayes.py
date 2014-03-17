import math
import string

"""
This class implements the naive bayesian classifier
"""


class BinaryNaiveBayesian(object):

    universe = dict()
    classes_document_count = dict()

    def __init__(self, datasource, stopwords_source=None):
        words = set()
        stopwords = set()
        if stopwords_source:
            with open(stopwords_source, 'r') as data:
                for line in data:
                    stopwords = stopwords.union(set([line.strip().lower()]))
        with open(datasource, 'r') as data:
            for line in data:
                _, score, sentence = line.split('|')
                class_number = int(math.floor(float(score) * 5))
                new_words = sentence.split(' ')
                new_words = set(new_words)  # remove duplicate words
                for word in new_words:
                    if not word in stopwords:
                        if word.strip() not in string.punctuation:
                            word = word.strip().lower()
                            # We keep a count of total number of document of a given class
                            self.classes_document_count[class_number] = self.classes_document_count.get(class_number, 0) + 1
                            # We keep a total count of documents of this class that have this word
                            if word in self.universe:
                                if class_number in self.universe[word]:
                                    self.universe[word][class_number] = self.universe[word][class_number] + 1
                                else:
                                    self.universe[word][class_number] = 1
                            else:
                                self.universe[word] = {
                                    class_number: 1
                                }

    def get_document_words(self, document):
        word_set = set()
        words = document.split(' ')
        for word in words:
            if word in self.universe:
                word_set.add(word)
        return word_set

    def get_proba_word_appears_in_class(self, word, class_number):
        """
        This method gives the probability that a word (i.e. document)
        comes from a given class, using naive bayesian method
        i.e. number of sentences of this classe that have the word on number of sentences of this class
        """
        small_alpha = 0.00001
        if word in self.universe:
            if class_number in self.universe[word]:
                documents_with_word_in_class = float(self.universe[word][class_number])
                documents_in_class = float(self.classes_document_count.get(class_number, 0))
                if documents_in_class:
                    return documents_with_word_in_class / documents_in_class
        return small_alpha

    def get_class_proba(self, class_number):
        """
        This method scores a given class given the input data
        It returns the number of documents appearing in class over
        sum of documents appearing in all classes
        """
        total_document_count = sum(self.classes_document_count[class_num] for class_num in self.classes_document_count)
        return float(self.classes_document_count.get(class_number, 0)) / float(total_document_count)

    def get_proba_document_in_class(self, document, class_number):
        """
        This method applies naive bayesian
        X document, Yi classes
        P(document X in class Yi) = P(Yi | X) = P(X | Yi) * P(Yi) / P(X)
        We actually ignore P(X) since we just want to compare probabilities and X doesn't change
        We want class for which P(X | Yi) * P(Yi) is maximal

        P(X, Yi) = product of P(xj | Yi) where xj words in X (independance)

        P(xj | Yi) = n(xj, Yi) / n(Yi) = number of documents in class with word / number of documents in class
        """
        document_words = self.get_document_words(document)
        P_X_Yi = -1
        for word in document_words:
            if P_X_Yi == -1:
                P_X_Yi = float(self.get_proba_word_appears_in_class(word, class_number))
            else:
                P_X_Yi = P_X_Yi * float(self.get_proba_word_appears_in_class(word, class_number))
        P_Yi = self.get_class_proba(class_number)
        P_Yi_X = P_X_Yi * P_Yi
        return P_Yi_X

    def get_document_class(self, document):
        class_result = -1
        max_proba = 0
        for class_number in range(5):
            proba = self.get_proba_document_in_class(document, class_number)
            if proba > max_proba:
                max_proba = proba
                class_result = class_number
        return class_result

    def get_bayes_results(self, test_source):
        success = 0
        count = 0
        with open(test_source) as test_data:
            for line in test_data:
                count += 1
                _, score, sentence = line.split('|')
                real_class = int(math.floor(float(score) * 5))
                bayesian_class = naive_bayes.get_document_class(sentence)
                if real_class == bayesian_class:
                    success += 1
        return float(success) / float(count)


if __name__ == '__main__':
    naive_bayes = BinaryNaiveBayesian('data/training_data/training.data')
    print naive_bayes.get_bayes_results('data/test_data/test.data')
