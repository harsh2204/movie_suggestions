import re
from pprint import pprint
import pandas as pd
jokes = []
for i in range(1, 101):
    pattern = '<!--begin of joke -->([\s\S]*)<!--end of joke -->'
    with open(f"init{i}.html", 'r') as f:
        contents = f.read()
        split = re.split(pattern, contents)
        joke = split[1:-1][0]
        # split is an array of 3 elements so we splice off the first and the last element. -> split[1:-1]
        # And then we access the remaining element in the array with -> [0]
        joke = re.sub('<[^<]+?>', '', joke)
        # Remove any <p> tags as they essentially have the same function as \n. -> .replace('<P>', '')
        joke = f"Joke #{i}:" + joke
        jokes.append({'number': i, 'text': joke})        

# Writing the array of jokes to a csv file
pd.DataFrame(jokes).to_csv("jokes.csv", index=False)
