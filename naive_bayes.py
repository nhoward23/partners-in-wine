import utils 
import numpy
import math
import random
import copy 
import tabulate
import operator
from random import choices 

def get_random_sample(table, n):
    """Gets n random instances from table and returns the index and instance of each table."""

    random_indices = []
    random_instances = [] 

    # get n unique random indices between [0, n)
    while len(random_indices) != n: 
        index = random.randrange(utils.count_instances(table))
        if index in random_indices:
            pass
        else:
            random_indices.append(index)

    # get list of instances based on the indexes
    for index in random_indices:
        random_instances.append(copy.deepcopy(table[index]))

    return random_indices, random_instances

def gaussian(x, mean, sdev):
  first, second = 0, 0
  if sdev > 0:
      first = 1 / (math.sqrt(2 * math.pi) * sdev)
      second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
  return first * second

def get_gaussian_posteriors(table, test_instance, indicies):
    posteriors = [] 

    for i in indicies:
        # calculate the gaussian probability
        att_vals = utils.get_column(table, i)
        mean = numpy.mean(att_vals)
        stdev = numpy.std(att_vals)
        prob = gaussian(test_instance[i], mean, stdev)
        posteriors.append(prob)
    
    #print("posteriors", posteriors)
    return posteriors

def get_posteriors_table(class_tables, test_instance, indicies):
    posteriors = []
    for table in class_tables:
        posteriors.append(get_gaussian_posteriors(table, test_instance, indicies))
    return posteriors

def get_prior_probabilities(table, classification_index):
    # group by classifications
    # get count of each table / count of table
    class_names, class_tables = utils.group_by(table, classification_index)
    probabilities = []
    count = utils.count_instances(table)

    for class_table in class_tables:
        # instances with given class
        class_count = utils.count_instances(class_table)

        # append probabilities
        probabilities.append(class_count/count)

    return class_names, probabilities, class_tables

def get_class_probabilities(priors, posteriors): 

    # for each prior multiply it by the product of the posteriors
    probabilities = []
    for i, prior in enumerate(priors):
        #probabilities.append(round(prior * numpy.prod(posteriors[i]), 4))
        probabilities.append(prior * numpy.prod(posteriors[i]))
    
    # return a parallel array of the probabilites for each class
    return probabilities

def get_classification(class_probabilities, classes):
    '''return the classification with the highest probability'''

    # get index of max probability and return class at index since its parrallel array
    max_prob = max(class_probabilities)
    index = class_probabilities.index(max_prob)
    classification = classes[index]
    return classification

def naive_bayes_classify_with_guassian(test_instance, table, class_index, attributes_indicies):
    # get the priors
    classes, priors, class_tables = get_prior_probabilities(table, class_index)
    #print("Classes: ", classes)
    #print('priors: ', priors)

    # get the posteriers for each attribute 
    posteriors = get_posteriors_table(class_tables, test_instance, attributes_indicies)
    #print("posteriors", posteriors)
    
    # calculate probabilites
    class_probabilities = get_class_probabilities(priors, posteriors)
    #print('class probabilities', class_probabilities)

    # get the highest probability and that is the class
    classification = get_classification(class_probabilities, classes)
    #print(classification)

    return classification

def guassian_naive_bayes(train_set, test_set, class_index, attribute_indicies):

    # classify each instance in test set and create parallel array of predictions and actuals
    predicted = []
    actual = [] 
    for instance in test_set:
        classification = naive_bayes_classify_with_guassian(instance, train_set, class_index, attribute_indicies)
        predicted.append(classification)
        actual.append(instance[class_index])

    return predicted, actual

def random_subsampling(k, table, class_index, predictive_indicies):
    """Uses random subsampling k times to predict instances"""

    naive_accuracies = []

    # for k times, randomize table and compute holdouts and classify with naive bayes
    for _ in range(k):
        # make a copy of the table to perform subsampling on so to not change the orginal table too much
        table_copy = copy.deepcopy(table)

        # randomize the table_copy and split 2:1
        train_set, test_set = compute_holdout_partitions(table_copy)

        # classify the test set by using naive bayes
        naive_predictions, naive_actuals = guassian_naive_bayes(train_set, test_set, class_index, predictive_indicies)

        # compute the accuracy of predictions for naive bayes
        naive_accuracy = compute_accuracy(train_set, class_index, naive_predictions, naive_actuals)
        
        # add accuacies to the lists keeping track of accuracies for regression and knn
        naive_accuracies.append(naive_accuracy)

    # now get the average accuracies for both classifiers
    avg_naive_acc = sum(naive_accuracies)/len(naive_accuracies)

    # get the standard error
    naive_std_err = 1 - avg_naive_acc
    return avg_naive_acc, naive_std_err

def compute_holdout_partitions(table):
    """Randomizes table and does a 1/3 split"""
    randomized = table[:] # shallow copy
    n = len(randomized)
    for i in range(n):
        # generate a random index to swap with i
        rand_index = random.randrange(0, n) # [0, n)
        # task: do the swap
        randomized[i], randomized[rand_index] = randomized[rand_index], randomized[i]

    # compute the split index
    split_index = int(2 / 3 * n)
    train_set = randomized[:split_index]
    test_set = randomized[split_index:]
    return train_set, test_set

def compute_accuracy(table, index, predicted, actual):
    """Computes the accuracy of predicted versus actual by taking the accuracy of each class and getting the
    average of these accuracies."""
    # list of accuracies 
    accuracies = []

    # get the classes 
    classes = list(set(utils.get_column(table, index)))

    # For each class, determine the accuracy
    for label in classes: 
    
        true_pos = 0 
        false_pos = 0
        false_neg = 0
        true_neg = 0

        # do the confusion matrix
        for i in range(len(predicted)):
            # when actual is 1 and predict is 1 - TP
            if actual[i] == label and predicted[i] == label: 
                true_pos += 1
            # when actual is not 1 and predicted is 1 - FP
            elif actual[i] != label and predicted[i] == label: 
                false_pos += 1
            # when actual is 1 and predicted is not 1 - FN
            elif actual[i] == label and predicted[i] != label:
                false_neg += 1
            # when actual is not 1 and predicted is not 1 - TN
            elif actual[i] != label and predicted != label: 
                true_neg += 1

        # see if it is an empty label (class isn't in the actual)s
        if label not in actual:
            pass
        else: 
            # compute the accuracy - TP + TN / P + N, and add it to list 
            accuracy = (true_pos + true_neg) / len(predicted)
            accuracies.append(accuracy)
            
    # get the average of accuracies
    avg_accuracy = sum(accuracies)/len(accuracies)
    return avg_accuracy



table = utils.read_table('red_wine_quality.csv')
header = table[0]
table = table[1:]

# lets just test with first two
indices = [0,3,5,6]

# random sample
random_instance = table[1:2]

print(header.index("quality"))

predicted, actual = guassian_naive_bayes(table, random_instance, header.index("quality"), indices)


random_indices, random_instances = get_random_sample(table, 5)


predicted, actual = guassian_naive_bayes(table, random_instances, header.index("quality"), indices)

print("predicted", predicted)
print("actual", actual)

# STEP 3 from PA4: Random SubSampling 
random_avg_accuracy, random_stderr = random_subsampling(5, table, header.index("quality"), indices)
print("=" * 76)
print("STEP 3: Predictive Accuracy of Naive Bayes with Guassian")
print("=" * 76)
print("Random Subsample (k=10, 2:1 Train/Test)")
print("Naive Bayes: accuracy = %0.2f, error rate = %0.2f" %(random_avg_accuracy, random_stderr))
print()