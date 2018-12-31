import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

#CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
#the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

#create model                                                               Precision@k=5
# model = LightFM(loss='logistic')  # Logistic regression                   -> 6.80942237377
# model = LightFM(loss='bpr')       # Bayesian Persionalized  Ranking       -> 6.50963634253
model = LightFM(loss='warp')        # Weighted Approximate-Rank Pairwise    -> 8.52248370647
# model = LightFM(loss='warp-kos')  # K'th order statistic loss             -> 7.32334181666
#train model
model.fit(data['train'], epochs=30, num_threads=2)

test_precision = precision_at_k(model, data['test'], k=5).mean()
print(test_precision*100)