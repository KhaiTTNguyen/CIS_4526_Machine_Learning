import sys
import csv
import numpy
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
numpy.set_printoptions(threshold=sys.maxsize)


def compute_error(test_y, pred_y):
    comparison = (test_y == pred_y)
    similar_count = 0

    for i in range(len(comparison)):
        if (comparison[i] == True):
            similar_count = similar_count + 1 
            
    return 1 - similar_count/len(comparison)


def main():
    
    '''
    Get the first command line argument of the program.
    For example, sys.argv[1] could be a string such as 'breast_cancer.csv' or 'titanic_train.csv'
    '''
    szDatasetPath = sys.argv[1]
	# Comment out the following line and uncomment the above line in your final submission
    # szDatasetPath = 'titanic_train.csv'

    '''
    Read the data from the csv file
    listColNames[j] stores the jth column name
    listData[i][:-1] are the features of the ith example
    listData[i][-1] is the target value of the ith example
    '''
    listColNames = [] # The list of column names
    listData = [] # The list of feature vectors of all the examples
    nRow = 0
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if 0 == nRow:
                listColNames = row
            else:
                listData.append(row)
            nRow += 1

    '''
    Scan the data and store the unique values of each column.
    listColUniqueVals[j] stores a list of unique values of the jth column
    '''
    listColUniqueVals = [[] for i in range(len(listColNames))]
    for example in listData:
        for i in range(len(example)):
            if example[i] not in listColUniqueVals[i]:
                listColUniqueVals[i].append(example[i])
    
    # -------------------------Part1: Compute the training error of a one-level decision tree--------------------------
    # List of errors, 1 error / feature
    features_errors = []
    for col in range(len(listColNames)-1):
        
        current_error = 0
        
        # given the target only has 2 unique values
        classify_array = numpy.zeros((len(listColUniqueVals[col]),2)) 
        
        for row in range(len(listData)):
            
            for index_of_uniqueVal in range(len(listColUniqueVals[col])):        
                if listData[row][col] == listColUniqueVals[col][index_of_uniqueVal]:
                    if(listData[row][-1] == "1" or listData[row][-1] == "recurrence-events"):
                        classify_array[index_of_uniqueVal][1] += 1
                        
                    else:
                        classify_array[index_of_uniqueVal][0] += 1
                        
        for l in range(len(listColUniqueVals[col])):
            current_error += min(classify_array[l])
            
        features_errors.append(current_error/len(listData))
    
    for col in range(len(listColNames)-1):
        print("Feature", col, "has error:", features_errors[col])
    
    # ------------------Part2: Construct a full decision tree on the dataset and compute the training error-------------------
    # Convert strings to integers
    feature_data_matrix = []
    targetColumn = []
    encoding = OrdinalEncoder()
    listData_encoded = encoding.fit_transform(listData)

    # random data
    numpy.random.shuffle(listData_encoded)
    sample_limit = int(len(listData) * 4 / 5)
    
    for i in range(len(listData_encoded)):
        feature_data_matrix.append(listData_encoded[i][:-1])
        targetColumn.append(listData_encoded[i][-1])
    
    # Initialize & construct tree using sklearn
    pred_tree = tree.DecisionTreeClassifier()
    pred_tree = pred_tree.fit(feature_data_matrix[:sample_limit],targetColumn[:sample_limit])
    
    # Predict & calculate error
    pred_targetColumn = pred_tree.predict(feature_data_matrix[sample_limit:])
    error = compute_error(targetColumn[sample_limit:],pred_targetColumn)
    print("Error for full decision tree:", error)
    
    # Display tree    
    tree.plot_tree(pred_tree.fit(feature_data_matrix[:sample_limit],targetColumn[:sample_limit]))

    
    return None

if __name__ == '__main__':

    main()