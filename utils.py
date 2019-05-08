# utils.py
# functions we use almost every assignment
import csv
import tabulate
import numpy 
import math

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
    
def knn_random_subsampling(tablename, attribute_indices, prediction_index, k):
    prediction_index_values = []
    actuals = []
    classes = []
    training = utils.normalize_instances(tablename, attribute_indices)
    print("===========================================")
    print("STEP 2: k=5 Nearest Neighbor MPG Classifier")
    print("===========================================")
    for instance in tablename:
        prediction_index_values.append(int(instance[prediction_index]))
    for index, row in enumerate(training):
        row.append(prediction_index_values[index])
    for i in range(5):
        random_index = random.randint(0, len(tablename)-1)
        random_instance = tablename[random_index]
        clean_table = []
        for index, row in enumerate(tablename):
            v1 = training[index]
            v2 = training[random_index]
            row.append(utils.compute_distance(v1[:-1],v2[:-1]))
            clean_table.append(row)
        clean_table.sort(key=itemgetter(-1))
        clean_table = clean_table[:5]
        prediction = 0
        print(prediction_index_values)
        max_val = max(prediction_index_values)
        for x in range(max_val+1):
            count = 0
            for row in clean_table:
                if (float(row[prediction_index])) == x:
                    count = count + 1
            if count > max:
                max = count
                prediction = x
        actual = (random_instance[prediction_index])
        actuals.append(actual)
        classes.append(prediction)
        print("instance: " + str(random_instance))
        print("class: "+ str(prediction))
        print("actual: "+ str(actual))
    return actuals, classes

    