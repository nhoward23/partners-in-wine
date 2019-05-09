import utils
from operator import itemgetter
import random
import ensemble_classification as es

def main():
    print("HI")
    table = utils.read_table('red_wine_quality.csv')
    header = table[0]
    data = table[1:100]
    utils.knn_random_subsampling(data, 5, [3, 7, 10], 11, 5)
    #es.bootstrap_aggregation(data, 20, [3, 7, 10], 11, .7, 5)

if __name__ == "__main__":
    main()
