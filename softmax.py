import numpy as np
import math


class Softmax(object):
    nombre_class=5
    l=0.001 # LAMBDA parameter


    def infer_class(self, W, x):
        return np.argmax(self.softmax(W.dot(x)))


    def softmax(self, vector):
        # difficulte pour les double overflow
        v_max = np.max(vector)
        exp_sum = 0.0

        for v in vector:
            exp_sum += math.exp(v - v_max)
        


        return [math.exp(vector[i] - v_max)/exp_sum for i in range(self.nombre_class)]


    def train_step(self, W, x, x_class, alpha, beta):
        # a(k) = w(k).x
        a = W.dot(np.transpose(x))
        p = self.softmax(a)
        #print p 

        U= [ [0] for j in range(self.nombre_class)]

        for j in range(self.nombre_class):
            if  x_class == j:
                U[j] = (p[j]-1) * x
            else:
                U[j] = p[j] * x

        return np.add(alpha*W, -beta*np.array(U))


    def train_epoch(self, W, shuffled_set):
        
        for t, (observation_vector, observation_class) in enumerate(shuffled_set):
            T = (t+1.0)/10
            W = self.train_step(
                W,
                observation_vector,
                observation_class,
                1.0 - (1.0/T),
                1.0/(self.l*(T + 1.0))
            )

        return W


    def test_classifier(self, W, test_set):
        error = 0
        success = 0
        for (observation_vector, observation_class) in test_set:
            if self.infer_class(W, observation_vector) == observation_class:
                success += 1
            else:
                error += 1

        return error, success