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
    print(total_cluster_score)
    return total_cluster_score
    
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
    print("Random Instance: ", random_instance)
    print("Majority Classification: ", majority_classification)
    
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

    
