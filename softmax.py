import numpy as np
import math


class Softmax(object):
    nombre_class= 5
    l=1


    def infer_class(self, W, x):
        return np.argmax(W.dot(x))


    def softmax(self, vector, index):
        exp_sum = 0.0

        for v in vector:
            exp_sum += math.exp(v)
        
        if exp_sum == 0:
            import ipdb
            ipdb.set_trace()

        return math.exp(vector[index])/exp_sum


    def train_step(self, W, x, x_class, alpha, beta):
        
        a = W.dot(np.transpose(x))
        a_prob = [self.softmax(a, i) for i in range(self.nombre_class)]
        a_class = np.argmax(a_prob)
        
        u = []

        for j in range(self.nombre_class):
            if x_class == float(i):
                if a_class != j:
                    # right
                    u.append(x*(a_prob[j] - 1))
                else:
                    # false
                    u.append(x*a_prob[j])
            else:
                if a_class != j:
                    # false
                    u.append(x*a_prob[j])
                else:
                    # right
                    u.append(x*(a_prob[j] - 1))
        return np.add(alpha*W, beta*np.array(u))


    def train_epoch(self, W, shuffled_set):
        
        for t, (observation_vector, observation_class) in enumerate(shuffled_set):
            W = self.train_step(
                W,
                observation_vector,
                observation_class,
                1.0 - (1.0/(t+1.0)),
                1.0/(self.l*((t+1) + 1.0))
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