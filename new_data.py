import numpy as np
# from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import pandas as pd
from pprint import pprint
from scipy import sparse
from sklearn.model_selection import train_test_split

#CHALLENGE part 1 of 3 - write your own fetch and format method for a different recommendation
#dataset. Here a good few https://gist.github.com/entaroadun/1653794 
#And take a look at the fetch_movielens method to see what it's doing 
#

def fetch_data():
    # Grab stuff from merged and jokes.
    # jokes.csv has all the labels
    # Grab labels first
    df = pd.read_csv("jester_dataset/jokes.csv")
    jokes = df.to_dict('split')
    jokes = [data[-1] for data in jokes['data']]
    # merged.csv has all the data
    arr = np.genfromtxt('jester_dataset/jester-data-1.csv',delimiter=',')
    # arr = np.genfromtxt('jester_dataset/merged.csv',delimiter=',')
    arr = np.delete(arr, 0, axis=1)
    arr = sparse.csr_matrix(arr)
    # print(arr.shape)
    # arr = np.transpose(arr)
    # print(arr.shape)
    data_train, data_test = train_test_split(arr, test_size=0.20, random_state=42)
    # # removes the first element    
    data = {'item_labels':jokes, 'train': data_train, 'test': data_test}
    return data
#fetch data and format it
# data = fetch_movielens(min_rating=4.0)
data = fetch_data()
#print training and testing data
# print(repr(data['train']))
# print(repr(data['test']))

# print(repr(data['item_labels'][data['train'][0].indices]))

#CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
#the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

#create model
model = LightFM(loss='warp')    # WARP = Weighted Approximate-Rank Pairwise
print("Training!")
model.fit(data['train'], epochs=30, num_threads=2)
print("Finding Recommendations!")

#CHALLENGE part 3 of 3 - Modify this function so that it parses your dataset correctly to retrieve
#the necessary variables (products, songs, tv shows, etc.)
#then print out the recommended results 

def sample_recommendation(model, data, user_ids):
    # TODO FIGURE THIS SHIT OUT!
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    print(n_users, n_items)
    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        # known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # The way this works is data['train'] is a huge matrix with all the training 
        # data based on each user's preference based on a user_id. .tocsr() method 
        # allows you to directly access the movie suggestions based on the user id using 
        # the array accessing method. indices is the property that allows you to access 
        # the indices corresponding to the indices of the movie names in the data['item_labels']. 
        # This returns an array of movies in the end despite it accessing the numpy array 
        # using a matrix array (1D).

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        top_scores = np.argsort(-scores)[:3]
        print(repr(scores))
        #rank them in order of most liked to least
        top_scores = scores[:3]
        for x in top_scores.tolist():
            print(x)
        # Argsort returns the sorted indices rather than a sorted array. -scores is just a 
        # way to sort it in the descending order.

        #print out the results
        print(f"User {user_id}")
        # print("\tKnown positives:")

        # for x in known_positives[:3]:
        #     print(f"\t\t{x}")

        # print("\tRecommended:")

        # for x in top_items[:3]:
        #     print(f"\t\t{x}")
            
# sample_recommendation(model, data, [3, 25, 450])
