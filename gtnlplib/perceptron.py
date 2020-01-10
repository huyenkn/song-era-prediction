from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector
from gtnlplib import clf_base
from gtnlplib import constants
from gtnlplib.constants import OFFSET

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    #raise NotImplementedError

    #f_xy = defaultdict(float)
    #f_xy_hat = defaultdict(float)
    
    y_hat,_ = clf_base.predict(x,weights,labels)

    update = defaultdict(float)
    if y != y_hat:
        #for word in x:
            #f_xy[((y, word))] = x[word]
            #f_xy_hat[((y_hat, word))] = x[word]
        f_xy = make_feature_vector(x,y)
        f_xy_hat = make_feature_vector(x,y_hat)
       
        for key in f_xy:
            update[key] = f_xy[key] - f_xy_hat.get(key, 0)
        for key in f_xy_hat:
            update[key] = -(f_xy_hat[key] - f_xy.get(key, 0))

    #if all(value == 0 for value in update.values()):
        #update = {}
    #else:
        #update[((y,constants.OFFSET))] = 1
        #update[((y_hat,constants.OFFSET))] = -1
    #update = {key: (f_xy[key] - f_xy_hat.get(key, 0)) for key in f_xy.keys()}
    return update

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            update = perceptron_update(x_i,y_i,weights,labels)
            for key in update:
                weights[key] += update[key] #+ weights.get(key, 0)
                if update[key] == 0:
                    del my_dict[key]
            #raise NotImplementedError
        weight_history.append(weights.copy())
    return weights, weight_history

