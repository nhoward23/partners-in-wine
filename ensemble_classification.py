import utils
import operator

def bootstrap_aggregation(table, num_class, attribute_indices, classification_index, min_acc, k):
    # runs classifiers over instances in the test set to produce predictions
    full_test = utils.generate_test_set(table)
    test_classifications = utils.get_classifications(full_test, classification_index)
    test = utils.normalize_instances(full_test, attribute_indices)
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
            final_prediction = utils.generate_weighted_majority_prediction(predictions, accuracies, test_classifications)
            full_sample = full_test[index]
            actual_classification = full_sample[classification_index]
            if (final_prediction == actual_classification):
                correct += 1
            #knn_print(actual_classification, final_prediction, sample)
        accuracy = correct/len(test)
        accuracy_print(accuracy)


def accuracy_print(accuracy):
    print("============================================================================")
    print("Predictive Accuracy of Aggregate Bootstrap Using KNN Classifiers")
    print("============================================================================")
    print("Bagging (k=5, 2:1 Remainder/Test")
    print("accuracy: "+str(accuracy))
    print("error rate: "+str(1-accuracy))

def classifier_prediction(training, attribute_indices, classification_index, test, k):
    # generates a prediction for an individual classifier
    classification_index_values, new_training = utils.knn_prep_training(training, attribute_indices, classification_index)
    clean_table = []
    for index, _ in enumerate(training):
        v1 = new_training[index]
        v2 = test
        new_row = training[index] + [utils.compute_distance(v1[:-1],v2)]
        clean_table.append(new_row)
    clean_table.sort(key=operator.itemgetter(-1))
    clean_table = clean_table[:k]
    prediction = utils.knn_majority_vote(classification_index_values, clean_table, classification_index)
    return prediction

def select_approved_training_sets(table, num_class, attribute_indices, classification_index, min_acc, k):
    # uses accuracy results of each training set to select only the ones that meet minimum accuracy constraints
    ensemble_train = []
    ensemble_accuracy = []
    for _ in range(num_class+1):
        training, validation = utils.bagging(table)
        accuracy = ensemble_performance(training, validation, attribute_indices, classification_index, k)
        if accuracy >= min_acc:
            ensemble_train.append(training)
            ensemble_accuracy.append(accuracy)
    return ensemble_train, ensemble_accuracy

def ensemble_performance(training, validation, attribute_indices, classification_index, k):
    # measures the performance of a training set against a validation set
    partial_validation = utils.normalize_instances(validation, attribute_indices)
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