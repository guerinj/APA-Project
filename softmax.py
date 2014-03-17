import numpy as np
import math


#
# La classe softmax fournit les methodes d'apprentissage et de test
# pour un softmax a 5 taches. La taille des vecteurs appris est indifferente
# W doit etre une matric 5xN
#
class Softmax(object):
    nombre_class=5
    l=0.01 # LAMBDA parameter


    def infer_class(self, W, x):
        return np.argmax(self.softmax(W.dot(x)))

    #
    # Calcul les probabilites softmax sur les composantes de vector
    #
    def softmax(self, vector):
        # difficulte pour les double overflow
        v_max = np.max(vector)
        exp_sum = 0.0

        for v in vector:
            exp_sum += math.exp(v - v_max)
        
        return [math.exp(vector[i] - v_max)/exp_sum for i in range(self.nombre_class)]


    #
    # Realise une etape d'apprentissage du classificateur a 5 taches
    #
    # W une matrice 5xn 
    # x un vecteur de l'espace de travail (soit Rn avec n le nombre de mots differents)
    # x_class la class de x
    # alpha et beta les coeffs
    def train_step(self, W, x, x_class, alpha, beta):
        # a(k) = w(k).x
        a = W.dot(np.transpose(x))
        p = self.softmax(a)
        #print p 

        # On va calculer U qui represente le gradient + la correction de poids ||w||^2
        U= [ [0] for j in range(self.nombre_class)]

        for j in range(self.nombre_class):
            #si on traitre la class de x : (p-1)x
            if  x_class == j:
                U[j] = (p[j]-1) * x
            # sinon p*x
            else:
                U[j] = p[j] * x

        return np.add(alpha*W, - beta*np.array(U))

    #
    # Realise une etape d'apprentissage
    #
    def train_epoch(self, W, shuffled_set):
        T=2
        for t, (observation_vector, observation_class) in enumerate(shuffled_set):
            T = t+1
            #if t % 100 == 0:
            #    T += 1

            W = self.train_step(
                W,
                observation_vector,
                observation_class,
                1,#1.0 - 1.0/T,  # tend vers 1 pour stabiliser le classifier
                0.3 #1.0/(self.l*(T+1.0)) # tend vers 0
            )

            # 1 / 0.1 pour alpha et beta donne environ 37% de reussite (tres bien)

        return W

    #
    # Fonction de test
    #
    def test_classifier(self, W, test_set):
        error = 0
        success = 0
        for (observation_vector, observation_class) in test_set:
            if self.infer_class(W, observation_vector) == observation_class:
                success += 1
            else:
                error += 1

        return error, success