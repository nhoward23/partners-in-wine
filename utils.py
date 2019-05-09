# utils.py
# functions we use almost every assignment
import csv
import tabulate
import numpy 
import math
import random
import operator

def write_table(header, table, filename):
    with open(filename, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        
        if header != "": 
            writer.writerow(header)
    
        for row in table:
            writer.writerow(row)

def read_table(filename):
    table = []
    
    with open(filename) as csv_file: 
        reader = csv.reader(csv_file, delimiter=';')
        for i, row in enumerate(reader):
            convert_to_numeric(row)
            table.append(row)
    return table

def convert_to_numeric(values):
    """Converts the numerical data types to floats in your data set"""
    for i in range(len(values)):
        try:
            numeric_val = float(values[i])
            values[i] = numeric_val
        except ValueError:
            pass

def count_instances(table):
    """return the number of instances in the dataset."""
    count = 0
    for row in table:
        count += 1
    return count

def get_column(table, index): 
    """ Return the column of values based on the index. """
    column = [] 
    for row in table: 
        if row[index] != 'NA':
            column.append(row[index])
    return column

def remove_instances(table):
    """Returns a new table that removes all instances with NA attributes."""
    new_table = []
    for row in table:
        if 'NA' not in row: 
            new_table.append(row)
    return new_table

def get_frequencies(table, column_index):
    """Get the count of a each unique instance of an attribute"""
    column = sorted(get_column(table, column_index))
    values = []
    counts = []

    for value in column: 
        if value not in values:
            values.append(value)
            #first time we have seen this value
            counts.append(1)
        else: 
            # weve seen it before and hte list is sorted 
            counts[-1] += 1 

    return values, counts

def pretty_print(grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=", ")
        print()
    print()

def compute_regression_line(x, y):
    # Compute the mx+b line 
    mean_x = numpy.mean(x)
    mean_y = numpy.mean(y)
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / \
            sum([(x[i] - mean_x) ** 2 for i in range(len(x))]) 
    b = mean_y - m * mean_x
    return m, b

def predict(m, xs, b): 
    ys = []
    for x in xs:
        ys.append(((m*x)+b))
    return ys

def group_by(table, attribute_index):
    group_names = sorted(list(set(get_column(table, attribute_index))))

    # now we need as list of subtables 
    # each table correspinds to a value in group_names
    # parallel arrays
    groups = [[] for name in group_names]
    for row in table:
        # which group does it belong to?
        group_by_value = row[attribute_index]
        index = group_names.index(group_by_value)
        groups[index].append(row)
    return group_names, groups

def discretization(table, rating, cutoffs, attribute):
    """this takes a table and disretizes the avalues"""
    for row in table: 
        for i, cutoff in enumerate(cutoffs):
            if row[attribute] <= cutoff:
                row[attribute] = rating[i]
                break
        row[attribute] = rating[i]

def classify_instance(value, rating, cutoffs): 
    """This takes an value and returns the classification of value using cutoff and rating."""
    for i, cutoff in enumerate(cutoffs): 
        if value <= cutoff:
            return rating[i]

def compute_distance(v1, v2):
    """computes the distance between v1 and v2 using Eucildean distance"""
    assert(len(v1) == len(v2))
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def normalize_instances(tablename, attribute_indices):
    #normalizes all the attributes of each index you pass in
    pre_normalization = [[] for x in attribute_indices]
    post_normalize = []
    table = [[] for row in tablename]

    for instance in tablename:
        for index, attribute in enumerate(attribute_indices):
            pre_normalization[index].append(float(instance[attribute]))

    for row in pre_normalization:
        row = normalize(row)
        post_normalize.append(row)

    for row in post_normalize:
        for index, item in enumerate(row):
            table[index].append(item)

    return table

def normalize(single_attribute_values):
    #normalizes a list of values all of the same type of attribute
    normalized = []
    for x in single_attribute_values:
        x = (x-min(single_attribute_values)) / ((max(single_attribute_values) - min(single_attribute_values)) * 1.0)
        normalized.append(x)
    return normalized

def knn_random_subsampling(tablename, test_number, attribute_indices, classification_index, k):
    classification_index_values, training = knn_prep_training(tablename, attribute_indices, classification_index)
    for _ in range(test_number):
        random_index = random.randint(0, len(tablename)-1)
        random_instance = tablename[random_index]
        clean_table = []
        for index, row in enumerate(tablename):
            v1 = training[index]
            v2 = training[random_index]
            new_row = row + [compute_distance(v1[:-1],v2[:-1])]
            clean_table.append(new_row)
        clean_table.sort(key=operator.itemgetter(-1))
        clean_table = clean_table[:k]
        prediction = knn_majority_vote(classification_index_values, clean_table, classification_index)
        actual = random_instance[-1:]
        knn_print(actual, prediction, random_instance)

def knn_print(actual, prediction, test):
    print("instance: " + str(test))
    print("class: "+ str(prediction))
    print("actual: "+ str(actual))

def bootstrap_aggregation(table, num_class, attribute_indices, classification_index, min_acc, k):
    # runs classifiers over instances in the test set to produce predictions
    full_test = generate_test_set(table)
    test_classifications = get_classifications(full_test, classification_index)
    test = normalize_instances(full_test, attribute_indices)
    training, accuracies = select_approved_training_sets(table, num_class, attribute_indices, classification_index, min_acc, k)
    correct = 0
    if(len(training) == 0):
        print("ensemble is empty")
    else:
        for index, sample in enumerate(test):
            for train_set in training:
                predictions = []
                prediction = classifier_prediction(train_set, attribute_indices, classification_index, sample, k)
                predictions.append(prediction)
            final_prediction = generate_weighted_majority_prediction(predictions, accuracies, test_classifications)
            full_sample = full_test[index]
            actual_classification = full_sample[classification_index]
            if (final_prediction == actual_classification):
                correct += 1
            #knn_print(actual_classification, final_prediction, sample)
        accuracy = correct/len(test)
        print("accuracy: "+str(accuracy))

def classifier_prediction(training, attribute_indices, classification_index, test, k):
    # generates a prediction for an individual classifier
    classification_index_values, new_training = knn_prep_training(training, attribute_indices, classification_index)
    clean_table = []
    for index, _ in enumerate(training):
        v1 = new_training[index]
        v2 = test
        new_row = training[index] + [compute_distance(v1[:-1],v2)]
        clean_table.append(new_row)
    clean_table.sort(key=operator.itemgetter(-1))
    clean_table = clean_table[:k]
    prediction = knn_majority_vote(classification_index_values, clean_table, classification_index)
    return prediction

def select_approved_training_sets(table, num_class, attribute_indices, classification_index, min_acc, k):
    # uses accuracy results of each training set to select only the ones that meet minimum accuracy constraints
    ensemble_train = []
    ensemble_accuracy = []
    for _ in range(num_class+1):
        training, validation = bagging(table)
        accuracy = ensemble_performance(training, validation, attribute_indices, classification_index, k)
        if accuracy >= min_acc:
            ensemble_train.append(training)
            ensemble_accuracy.append(accuracy)
    return ensemble_train, ensemble_accuracy

def ensemble_performance(training, validation, attribute_indices, classification_index, k):
    # measures the performance of a training set against a validation set
    partial_validation = normalize_instances(validation, attribute_indices)
    total = 0
    correct = 0
    for index, line in enumerate(partial_validation):
        prediction = classifier_prediction(training, attribute_indices, classification_index, line, k)
        full_line = validation[index]
        actual_classification = full_line[classification_index]
        total += 1
        if prediction == actual_classification:
            correct += 1
    accuracy = correct/total
    return accuracy

def generate_majority_prediction(predictions, test_classifications):
    # generates a majority vote prediction given a list of predictions of a classifier
    prediction = 0
    max_classification_val = max(test_classifications)
    max_val = 0
    for x in range(max_classification_val+1):
        count = 0
        for prediction in predictions:
            if prediction == x:
                count = count + 1
        if count > max_val:
            max_val = count
            prediction = x
    return prediction


def generate_weighted_majority_prediction(predictions, accuracies, test_classifications):
    # generates a weighted majority vote prediction given a list of predictions of a classifier and that classifiers accuracy
    prediction = 0
    max_classification_val = max(test_classifications)
    max_val = 0
    for x in range(max_classification_val+1):
        count = 0
        for index, prediction in enumerate(predictions):
            if prediction == x:
                count = count + accuracies[index]
        if count > max_val:
            max_val = count
            prediction = x
    return prediction


def generate_test_set(table):
    instances = len(table)-1
    test = table[:int(1/3 * instances)]
    return test

def generate_remainder_set(table):
    instances = len(table)-1
    remainder = table[int(1/3 * instances):]
    return remainder

def bagging(table):
    instances = len(table)-1
    remainder = generate_remainder_set(table)
    training = []
    validation = []
    for _ in range(len(remainder)):
        random_instance_index = random.randint(0, len(remainder)-1)
        random_instance = remainder[random_instance_index]
        training.append(random_instance)

    for instance in remainder:
        for index, train in enumerate(training):
            if (instance == train):
                break
            elif (index == len(training)-1):
                validation.append(instance)

    return training, validation

def knn_majority_vote(classification_index_values, clean_table, classification_index):
    prediction = 0
    max_classification_val = max(classification_index_values)
    max_val = 0
    for x in range(max_classification_val+1):
        count = 0
        for row in clean_table:
            if (float(row[classification_index])) == x:
                count = count + 1
        if count > max_val:
            max_val = count
            prediction = x
    return prediction

def knn_prep_training(tablename, attribute_indices, classification_index):
    classification_index_values = get_classifications(tablename, classification_index)
    training = normalize_instances(tablename, attribute_indices)
    for index, row in enumerate(training):
        row.append(classification_index_values[index])
    return classification_index_values, training

def get_classifications(tablename, classification_index):
    classification_index_values = []
    for instance in tablename:
        classification_index_values.append(int(instance[classification_index]))
    return classification_index_values

    