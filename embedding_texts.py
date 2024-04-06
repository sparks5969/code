import pandas as pd
import json
import numpy as np
import openai
from openai import OpenAI
import os
import boto3
from get_secret import get_secret
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

tqdm.pandas()

# create a function to get the embeddings of a text
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


# import the data
data = pd.read_csv("example_sentences.csv")
# remove the rows if the example sentences are too short. say less than 5 words
data = data[data["example sentences"].apply(lambda x: len(x.split()) > 5)]

# retrieve the secret key and set it as an environment variable
key = get_secret()
key = json.loads(key)
os.environ['OPENAI_API_KEY'] = key['OPENAI_API_KEY']
client = OpenAI()

# in the df data, create a new column for embeddings
data['embedding'] = data["example sentences"].progress_apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

# export the data to a csv file
data.to_csv("example_sentences_embeddings.csv", index=False)

