import random
import math
import copy
import numpy
import matplotlib.pyplot as plt
import utils

def compute_distance(v1, v2):
    """computes the distance between v1 and v2 using Eucildean distance. Does not include the classification attribute."""
    #assert(len(v1) == len(v2))
    dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v2)-1)]))
    return dist

def select_init_centroids(k, table):
    """select k instances at random to use as the initial centroids.""" 
    init_centroids = []
    for i in range(k):
        instance_index = random.randrange(len(table))
        while table[instance_index] in init_centroids:
            instance_index = random.randrange(len(table))
        init_centroids.append(table[instance_index])
    return init_centroids

def append_distance(table, distances, point):
    """appends the distance between row in table and point to distances."""
    for i, row in enumerate(table):
        distance = compute_distance(row, point)
        distances[i].append(distance)
        
def find_centroid(distances):
    """gets the centroid index of smallest distance and appends it to distances."""
    for row in distances:
        cluster_index = row.index(min(row))
        row.append(cluster_index)

def combine_tables(table1, table2):
    """combines table 1 and table 2 together."""
    new_table = []
    assert(len(table1) == len(table2))
    for i, row in enumerate(table1):
        new_row = row + table2[i]
        new_table.append(new_row)
    return new_table

def compute_average(cluster, atts_range):
    """atts_range: is the range [0-att_range) in which to compute averages."""
    centroid = []
    for i in range(atts_range):
        column = utils.get_column(cluster, i)
        att_average = numpy.mean(column)
        centroid.append(round(att_average,2))
    return centroid
        
def recalculate_centroids(table, distances_table, att_list):
    # Combine and group by cluster 
    new_table = combine_tables(table, distances_table)
    cluster_index = len(new_table[0])-1
    group_names, groups = utils.group_by(new_table, cluster_index)
    
    # For each cluster
    new_centroids = []
    for cluster in groups:
        # get the average of each attribute
        new_centroid = compute_average(cluster, len(table[0]))
        new_centroids.append(new_centroid)
    return new_centroids
    
def compare_centroids(centroids1, centroids2):
    assert(len(centroids1) == len(centroids2))
    for i in range(len(centroids1)):
        if centroids1[i] != centroids2[i]:
            return False
    return True 

def k_means_clustering(k, table):
    # Select k objects in an arbitrary fashion. Use these as the initial set of k centroids.
    centroids = select_init_centroids(k, table)
    
    match = False
    while match == False: 
        
        # Compute the distances of each instance to each centroid
        distances_table = [ [] for i in range(len(table)) ]
        for centroid in centroids:
            append_distance(table, distances_table, centroid)

        # Find the biggest distance and assign instance to that centroid
        find_centroid(distances_table)

        # Recalculate the centroids by getting the mean of each cluster
        new_centroids = recalculate_centroids(table, distances_table, [])

        # Check to see if the centroids have converged
        match = compare_centroids(centroids, new_centroids)
        centroids = new_centroids

    # Now we know what instance belongs to what cluster
    # these are "parallel tables" 
    score = objective_function(table, distances_table, centroids)
    return score, distances_table, centroids
    
def objective_function(table, cluster_table, centroids):
    # Combine and group by cluster 
    new_table = combine_tables(table, cluster_table)
    cluster_index = len(new_table[0])-1
    group_names, groups = utils.group_by(new_table, cluster_index)
    
    # for each cluster, compute the sum of squares
    # add these to a cluster_total of all clusters
    total_cluster_score = 0
    for i, cluster in enumerate(groups):
        distances = []
        for row in cluster:
            distance = compute_distance(row, centroids[row[cluster_index]])
            distances.append(distance)
        total_cluster_score += sum(distances)
    # print(total_cluster_score)
    return total_cluster_score

    
def find_best_cluster(table, minimum, maximum):
    objective_scores = []
    x_axis = []
    for i in range(minimum, maximum+1):
        print(i)
        score, cluster_table, centroids = k_means_clustering(i, table)
        objective_scores.append(score)
        x_axis.append(i)
    
    plt.figure()
    plt.plot(x_axis, objective_scores)
    plt.show()

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

def majority_voting(cluster, classification_index):
    # get the frequency of the classfication index
    values, counts = utils.get_frequencies(cluster, classification_index)
    
    # get the biggest one 
    highest_freq_index = counts.index(max(counts))
    
    # return that classification
    return values[highest_freq_index]
    
def predict(random_instance, centroids, clusters, classification_index):
    distances = []
    for centroid in centroids:
        distance = compute_distance(random_instance, centroid)
        distances.append(distance)
    cluster_index = distances.index(min(distances))
    
    majority_classification = majority_voting(clusters[cluster_index], classification_index)
#     print("Random Instance: ", random_instance)
#     print("Majority Classification: ", majority_classification)
    return majority_classification

def k_means_classifier(train_set, test_set, class_index, centroids, clusters):

    # classify each instance in test set and create parallel array of predictions and actuals
    predicted = []
    actual = [] 
    for instance in test_set:
        classification = predict(instance, centroids, clusters, class_index)
        predicted.append(classification)
        actual.append(instance[class_index])

    return predicted, actual

def random_subsampling(k, table, class_index, centroids, clusters):
    """Uses random subsampling k times to predict instances"""
    naive_accuracies = []

    # for k times, randomize table and compute holdouts and classify with naive bayes
    for _ in range(k):
        # make a copy of the table to perform subsampling on so to not change the orginal table too much
        table_copy = copy.deepcopy(table)

        # randomize the table_copy and split 2:1
        train_set, test_set = compute_holdout_partitions(table_copy)

        # compute new cluster for each train set
        score, distances_table, centroids = k_means_clustering(6, train_set)
        # create a cluster table
        cluster_table = copy.deepcopy(train_set)
        for i, row in enumerate(cluster_table):
            cluster_table[i].append(distances_table[i][-1])
        # group by cluster 
        group_names, groups = utils.group_by(cluster_table, len(cluster_table[0])-1)
        # print(groups)

        # classify the test set by using naive bayes
        naive_predictions, naive_actuals = k_means_classifier(train_set, test_set, class_index, centroids, clusters)
        # print(naive_predictions)
        # print(naive_accuracies)

        # compute the accuracy of predictions for naive bayes
        naive_accuracy = compute_accuracy(train_set, class_index, naive_predictions, naive_actuals)
        
        # add accuacies to the lists keeping track of accuracies for regression and knn
        naive_accuracies.append(naive_accuracy)

    # now get the average accuracies for both classifiers
    avg_naive_acc = sum(naive_accuracies)/len(naive_accuracies)

    # get the standard error
    naive_std_err = 1 - avg_naive_acc
    return avg_naive_acc, naive_std_err


# table = utils.read_table('red_wine_quality.csv')
# header = table[0]
# table = table[1:]

        
# find_best_cluster(table, 3, 10)
#From plotting these objective functions scores, it appears that k=6 is an appropriate value.
#score, distances_table, centroids = k_means_clustering(7, table)
# create a cluster table
#cluster_table = copy.deepcopy(table)
#for i, row in enumerate(cluster_table):
#    cluster_table[i].append(distances_table[i][-1])
#utils.pretty_print(cluster_table)
#print(centroids)
# group by cluster
#group_names, groups = utils.group_by(cluster_table, len(cluster_table[0])-1)
#print(group_names)
#utils.pretty_print(cluster_table)

# predicting
#random_instance = [8.9,0.22,0.48,1.8,0.077,29,60,0.9968,3.39,0.53,9.4,6]
#predict(random_instance, centroids, groups, header.index("quality"))
#utils.pretty_print(table)

#for group in groups:
#    values, counts = utils.get_frequencies(group, header.index("quality"))
#    avg_att_vals = compute_average(group, len(group[0])-1)
#    print("=" * 60)
#    print("values: ", values)
#    print("counts: ", counts)
#    print("average vals:", avg_att_vals)
#    print("=" * 60)
    #utils.pretty_print(group)
    #values, counts = utils.get_frequencies(group, header.index("quality"))
    #print("=" * 60)
    #print("=" * 60)
    
    # we can get the average of each attribute and see how they differ for each group
    # total sulfur dioxide varies between each, appears lower the better
    # free sulfur dioxide is the same, less is higher quality
    # higher fixed acidity


# table = utils.read_table('red_wine_quality.csv')
# header = table[0]
# table = table[1:]
# # cluster the dataset to form 6 groups 
# score, distances_table, centroids = k_means_clustering(6, table)

# # create a cluster table
# cluster_table = copy.deepcopy(table)
# for i, row in enumerate(cluster_table):
#     cluster_table[i].append(distances_table[i][-1])

# # group by cluster 
# group_names, groups = utils.group_by(cluster_table, len(cluster_table[0])-1)


# random_avg_accuracy, random_stderr = random_subsampling(5, table, header.index("quality"), centroids, groups)
# print("=" * 76)
# print("Predictive Accuracy of K-Means Clustering with Random Subsampling")
# print("=" * 76)
# print("Accuracy = %0.2f, error rate = %0.2f" %(random_avg_accuracy, random_stderr))
# print()