from data_reader import *
import numpy as np
import math
from ipdb import launch_ipdb_on_exception
import ipdb
# class InformationMutuelle(object):

#     datasource = 'data/datasmall'
#     sentences = list()

#     def __init___(self):
#         with open(self.datasource, 'r') as data:
#             for line in data:
#                 _, score, sentence = line.split('|')
#                 self.sentences.append({'sentence': sentence, 'score': score})

with launch_ipdb_on_exception():
    datareader = DataReader('data/training_data/training.data', 'data/stopwords/stopwords.txt')
    sentences = list()

    with open('data/training_data/training.data', 'r') as data:
        for line in data:
            _, score, sentence = line.split('|')
            sentences.append({'sentence': sentence, 'score': float(score)})

    # empty vectors
    prob_observation = datareader.get_sentence_coordinates("")
    prob_class = np.zeros((2))    
    prob_class_observation = [datareader.get_sentence_coordinates("") for i in range(5) ]
   

    nombre_observations = len(sentences)
    nombre_observations_by_class = [0, 0]

    print "Scanning sentences"
    for sentence in sentences:
        sentence_class = int(math.floor(sentence['score']*2.0) - 1) # 0 ou 1
        sentence_vector = datareader.get_sentence_coordinates(sentence['sentence'])

        prob_observation = prob_observation + sentence_vector
        
        prob_class = prob_class + np.array([ 1 if (sentence_class == i ) else 0 for i in range(2)])
        
        prob_class_observation[sentence_class] = prob_class_observation[sentence_class] + sentence_vector
        
        nombre_observations_by_class[sentence_class] += 1
    
    print "Normalization"
    prob_observation = prob_observation / nombre_observations
    prob_class = prob_class / nombre_observations
    for i in range(2):
        prob_class_observation[i] = prob_class_observation[i] / nombre_observations_by_class[i]

    # Now we compute mutal information
    im_by_coordinate = list()
    print "Now we compute mutal information"
    for index, coordinate in enumerate(datareader.universe):
        
        im = 0
        for presence in range(2):
            for c in range(2):
                P_XY = prob_class_observation[c][index] if presence == 1 else 1 - prob_class_observation[c][index]
                P_Y = prob_class[c]
                P_X = prob_observation[index] if presence == 1 else 1 - prob_observation[index]
                if P_XY != 0 and P_X*P_Y != 0:
                    im += P_XY*math.log(P_XY/(P_X*P_Y))
        
        im_by_coordinate.append({"im" : im, "coordinate" : coordinate})
        print "Coordinate : %s IM : %s \n " % (coordinate, im)

    

    def compareCoordinate(a, b):
        if a["im"] > b["im"]:
            return 1
        elif a["im"] == b["im"]:
            return 0
        else: 
            return -1

    print "\n\n\n IM TOP 100"
    im_by_coordinate.sort(compareCoordinate)

    for index, coord in enumerate(im_by_coordinate):
        print "%s \t Word : %s \t IM : %s" % (len(im_by_coordinate) - index, coord["coordinate"], coord["im"])
