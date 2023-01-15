import sys, json
sys.path.append(sys.path[0]+'/utils/')
from datetime import datetime
from surprise import SVD
from surprise import NormalPredictor
from utils.Evaluator import Evaluator
from utils.MovieLens import MovieLens
import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

##### Add algorithms #####
# Throw in an SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

# Make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

##### Evaluation #####
result_dict = evaluator.Evaluate(True)

##### Save results as json file #####
exp_name = "exp_rand_svd"
date = datetime.today().strftime(format='%Y%M%d%H%m%s')
filename = './movielens/results/{}-{}.json'.format(exp_name, date)

with open(filename, 'w') as f:
     json.dump(result_dict, f)