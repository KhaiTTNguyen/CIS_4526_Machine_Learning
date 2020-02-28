# Note: this is just a template for PA 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged. Change the returned values of
# these functions so that they are consistent with the assignment instructions.
# In general, one will only need to add the code below the TO-DO statements to
# finish the assignment. Additional import statements can be included when needed.
#
# For the kNN classifier, one could use existing libraries to compute the pairwise
# Euclidean distances between the test and training data, as for-loops in Python
# are pretty slow. Other than that, the designs of all functions should be your
# original work.

import csv
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt 

# acc = compute_accuracy(test_y, pred_y)
# test_y is a (num_test,) label vector
# pred_y is a (num_test,) label vector
# acc is a float between 0.0 and 1.0
##
def compute_accuracy(test_y, pred_y):
    comparison = (test_y == pred_y)
    similar_count = 0

    for i in range(len(comparison)):
        if (comparison[i] == True):
            similar_count = similar_count + 1 
            
    return similar_count/len(comparison)

#--------------------------------------K-NN------------------------------------------#
# calculate Euclidean distance between 2 data points
def euclidean_distance(pointA, pointB):    
    return np.linalg.norm(pointA - pointB)

"""
    pred_y = test_knn(train_x, train_y, test_x, num_nn)
    
    train_y is a (num_train,) label vector,
    pred_y is a (num_test,) label vector
    num_nn is the number of nearest neighbors for classification.
"""
def test_knn(train_x, train_y, test_x, num_nn):
    
    num_test_data = len(test_x)
    temp_pred_y = np.zeros(num_test_data)
    # loop through testX
    # compute Euclidean with trainX
    # find K nearest
    # sort & get num_nn neighbors
    for i in range(len(test_x)):
        distance_to_trainX = []
        for j in range(len(train_x)):
            dist = euclidean_distance(test_x[i], train_x[j])
            distance_to_trainX.append((dist, j))
        
        distance_to_trainX.sort()
        neighbors = distance_to_trainX[:num_nn]
        
        # take majority
        # we use a Counter to find # of datapoints / label
        # most_common([n]) - return a list of the n most common elements
        num_each_type = Counter()
        for member_distance in neighbors:
            member_label = train_y[member_distance[1]]
            num_each_type[member_label] += 1

        vote_result = num_each_type.most_common(1)[0][0] # get name of most common type
        # label current test data ---> put in pred y[i]
        temp_pred_y[i] = vote_result;
        
    return temp_pred_y


#----------------------------------Pocket Algorithm------------------------------#
# produce label for dot product of w & x(i)
# OR 1 entry for pred_y
##
def sign_function(weight_vector, train_data):
    to_eval = np.dot(weight_vector[1:], train_data) + weight_vector[0]
    if to_eval < 0:
        return -1
    else:
        return 1
    
"""
    pred_y = test_pocket(w, test_x)
    
    w is a vector of learned perceptron weights, 
    pred_y is a (num_test,) +1/-1 label vector.
    
    # produce pred_y array
    # pred_y = test_pocket(w, test_x)
    # test_x is a (num_test, num_dims) data matrix

"""    
def test_pocket(w, test_x):
    pred_y = np.zeros(len(test_x))
    for i in range(len(test_x)):
        pred_y[i] = sign_function(w, test_x[i])
    
    return pred_y

"""
    w = train_pocket(train_x, train_y, num_iters)
    train_y is a (num_train,) +1/-1 label vector
    w is a vector of learned perceptron weights.

    # produce 1 perceptron weight vector
    # w = train_pocket(train_x, train_y, num_iters)
    # initialize a w_vector --> generate pred_y (using w & train_x)
    # find wrongly classified 
    # generate new_w --> generate new pred_y
    # compute accuracy for old pred_y & new pred_y
    # if increases ---> update w
    ##

"""
def train_pocket(train_x, train_y, num_iters):    
    weight_vector = np.zeros(len(train_x[0]) + 1) 
    for i in range(num_iters):
        current_weight = np.copy(weight_vector)
        for inputs,label in zip(train_x,train_y):    
            # find missclassified, generate current_weight
            predict_label = sign_function(current_weight,inputs)        
            if (predict_label != label):
                current_weight[1:] = current_weight[1:] + label * inputs
                current_weight[0] = current_weight[0] + label
                
        old_train_y = test_pocket(weight_vector, train_x)
        new_train_y = test_pocket(current_weight, train_x)
        old_acc = compute_accuracy(old_train_y, train_y)
        new_acc = compute_accuracy(new_train_y, train_y)
        
        if new_acc > old_acc:
            weight_vector = current_weight
    
    return weight_vector


# return my TU_ID
def get_id():
    return 'tuh23333'

# return array of converted label
def convert_label(interested_label, t_array):
    for i in range(len(t_array)):
        if t_array[i] == interested_label:
            t_array[i] = 1
        else:
            t_array[i] = -1
    return t_array


#-----------------------------------------------Main program--------------------------------------------#
def main():

    # Read the data file
    szDatasetPath = './letter-recognition.data' # Put this file in the same place as this script
    listClasses = []
    listAttrs = []
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            listClasses.append(row[0])
            listAttrs.append(list(map(float, row[1:])))

    # Generate the mapping from class name to integer IDs
    mapCls2Int = dict([(y, x) for x, y in enumerate(sorted(set(listClasses)))])

    # Store the dataset with numpy array
    dataX = np.array(listAttrs)
    dataY = np.array([mapCls2Int[cls] for cls in listClasses])
    
    
    # Split the dataset as the training set and test set
    nNumTrainingExamples = 15000
    trainX = dataX[:nNumTrainingExamples, :]
    trainY = dataY[:nNumTrainingExamples]
    testX = dataX[nNumTrainingExamples:, :]
    testY = dataY[nNumTrainingExamples:]

    #-------------------------------Assignment starts here------------------------------------------#

    # subsample 
    # num_train_list = [100, 1000, 2000, 5000, 10000, 15000]
    num_train = 100    
    pairs_to_sample = list(zip(trainX,trainY))

    random.shuffle(pairs_to_sample)

    # Take the first num_train elements of the randomized array
    t_trainX, t_trainY = zip(*pairs_to_sample[0:num_train])
    t_trainX = list(t_trainX)
    t_trainY = list(t_trainY)
    
    t_testX = testX.copy()
    t_testY = testY.copy()
    
    # test k-nn
    #k_list = [1,3,5,7,9]
    numNN = 3
    print("Test_k-nn")
    pred_y = test_knn(t_trainX, t_trainY, t_testX, numNN);
    print("With k =",numNN, "and num_train =", num_train) 
    print("Accuracy is: ", compute_accuracy(testY, pred_y))

    print()
    # test pocket
    print("Test_Pocket")
    # plot the curve (training accuracy v.s. iterations) to choose num_iters
    num_iteration = 500
    letter_num_tags = list(set(dataY))  # turn letter_tags to number_tags

    # loop through 26 letters
    print("Num_train =", num_train)
    for index in range(len(letter_num_tags)):
        t_trainY = convert_label(letter_num_tags[index], t_trainY.copy())
        t_testY = convert_label(letter_num_tags[index], testY.copy())

        # create a weight vector for thisletter
        t_weight = train_pocket(t_trainX, t_trainY, num_iteration)
        pred_y = test_pocket(t_weight, testX)
        acc = compute_accuracy(t_testY, pred_y)

        print("Letter",letter_num_tags[index], "has accuracy:", acc)

    return None

if __name__ == "__main__":
    main()