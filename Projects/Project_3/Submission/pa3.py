import csv
import numpy as np
import random
import sys

'''
The loss functions shall return a scalar, which is the *average* loss of all the examples
'''

'''
For instance, the square loss of all the training examples is computed as below:

def squared_loss(train_y, pred_y):

    loss = np.mean(np.square(train_y - pred_y))

    return loss
'''

# return average loss --> use np.mean() ???
# log(1 + exp(-train_y.pred_y))
# train_y = -1/1 vector, pred_y = W.X

def logistic_loss(train_y, pred_y):
    toMinimize = np.multiply(train_y, pred_y)
    log_loss = np.log(1 + np.exp(-toMinimize))
    return np.mean(log_loss)

def hinge_loss(train_y, pred_y):
    toMinimize = np.multiply(train_y, pred_y)
    hinge_loss = np.maximum(0, 1 - toMinimize)
    return np.mean(hinge_loss)

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''
def l1_reg(w):
    # take sum of absolute values 
    l1_loss = 0;
    for i in range(1, w.size):
        l1_loss += abs(w[i])
        
    return l1_loss

def l2_reg(w):
    return np.dot(w[1:], np.transpose(w[1:]))


# train_y is a (num_train,) +1/-1 label vector
# The complete training process should be implemented within the train_classifier function. 
# One may expect to perform sufficient number of iterations with gradient descent to update the weights in this function.

def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):
    
    # create pred_y --> goes into loss function
    
    # build expression for objective function
    
    # since the Objective function is determined only at run-time
    # compute gradient  partial derivative of each and aggregate them in a single gradient vector
    
    # w = w - learning_rate * deriv(loss function/w) -- for logistic loss only?
    
    weight_vector = np.random.rand(len(train_x[0]) + 1) # bias term included
    num_iters = 2000
    h = 0.0001 # numerical_differentiation 
    for i in range(num_iters):
        current_weight = np.copy(weight_vector)
        delta_weight = np.zeros(len(train_x[0]) + 1) # produce delta_weight to update weight w = w - delta_weight
        
        predict_y = test_classifier(current_weight,train_x)
        if(lambda_val):
            current_loss = loss(train_y, predict_y) + lambda_val*regularizer(current_weight)
        else:
            current_loss = loss(train_y, predict_y)
            
        
        for index in range(len(delta_weight)):
            temp_current_weight = np.copy(current_weight)
            temp_current_weight[index] = temp_current_weight[index] + h;
                
            temp_predict_y = test_classifier(temp_current_weight,train_x)
            
            # produce loss
            if(lambda_val):
                temp_loss = loss(train_y, temp_predict_y) + lambda_val*regularizer(temp_current_weight)
            else:
                temp_loss = loss(train_y, temp_predict_y)
            
            # partial differentiation
            delta_weight[index] = (temp_loss - current_loss) / h

        # update weight vector
        weight_vector = current_weight - learn_rate*delta_weight    # W = W - dl/dW    
           
    return weight_vector

# retrun pred_y 
# not a binary (-1/+1) vector 
# but the inner products of weights and feature values
def test_classifier(w, test_x):
    pred_y = np.zeros(len(test_x))
    for i in range(len(test_x)):
        pred_y[i] = np.dot(w[1:], test_x[i]) + w[0] 
    return pred_y

def normalize(trainX,testX):
    # Standardize the dataset
    dataX_Transposed = trainX.transpose()
    column = 0
    for row in dataX_Transposed:
        mean = np.mean(row)
        std = np.std(row)
        for index in range(len(trainX.transpose()[0])):
            trainX[index][column] -= mean
            trainX[index][column] /= std
        for index in range(len(testX.transpose()[0])):
            testX[index][column] -= mean
            testX[index][column] /= std
        column += 1


def compute_accuracy(test_y, pred_y):
    compute_pred_y = np.copy(pred_y)
    for j in range(len(compute_pred_y)):
        if(compute_pred_y[j] < 6):
            compute_pred_y[j] = -1
        elif (compute_pred_y[j] > 6):
            compute_pred_y[j] = 1

    comparison = (test_y == compute_pred_y)
    match_count = 0

    for i in range(len(comparison)):
        if (comparison[i] == True):
            match_count = match_count + 1 
            
    return (match_count/len(test_y))


def main():

    # Read the training data file
    szDatasetPath = 'winequality-white.csv'
    listClasses = []
    listAttrs = []
    bFirstRow = True
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if bFirstRow:
                bFirstRow = False
                continue
            if int(row[-1]) < 6:
                listClasses.append(-1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
            elif int(row[-1]) > 6:
                listClasses.append(+1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))

    dataX = np.array(listAttrs)
    dataY = np.array(listClasses)

    nNumTrainingSubset = int((len(dataX)/5))
    
    for i in range(5):
        if(i == 0):
            subdataX = np.split(dataX,[nNumTrainingSubset])
            subdataY = np.split(dataY,[nNumTrainingSubset])
            trainX = subdataX[1]
            trainY = subdataY[1]
            testX = subdataX[0]
            testY = subdataY[0]
        elif(i == 4):
            subdataX = np.split(dataX,[nNumTrainingSubset*i])
            subdataY = np.split(dataY,[nNumTrainingSubset*i])
            trainX = subdataX[0]
            trainY = subdataY[0]
            testX = subdataX[1]
            testY = subdataY[1]
        else:
            subdataX = np.split(dataX,[nNumTrainingSubset*i,nNumTrainingSubset*(i+1)])
            subdataY = np.split(dataY,[nNumTrainingSubset*i,nNumTrainingSubset*(i+1)])
            trainX = np.concatenate((subdataX[0],subdataX[2]),axis=0)
            trainY = np.concatenate((subdataY[0],subdataY[2]),axis=0)
            testX = subdataX[1]
            testY = subdataY[1]
        normalize(trainX,testX)
        
        # Logistic regression is a linear model trained with the logistic loss without the regularization term
        # Soft-margin SVM is (equivalent to) a linear model trained with the hinge loss with the l2 regularization term
        
        weight_vector_logistic = train_classifier(trainX,trainY,0.001,logistic_loss)
        weight_vector_SVM = train_classifier(trainX,trainY,0.1,hinge_loss,0.001, l2_reg)
        
        predict_y_log = test_classifier(weight_vector_logistic,testX)
        predict_y_SVM = test_classifier(weight_vector_SVM,testX)
        
        acc_log = compute_accuracy(testY, predict_y_log)
        acc_SVM = compute_accuracy(testY, predict_y_SVM)
        
        print(i,"th iteration")
        print("Accuracy is of Logistic Regression", acc_log)
        print("Accuracy is of SVM", acc_SVM)
    return None

if __name__ == "__main__":

    main()