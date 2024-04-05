import pandas as pd
import boto3
import json
import os
import numpy as np
import openai
from openai import OpenAI
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


# retrieve the secret key and set it as an environment variable
key = get_secret()
key = json.loads(key)
os.environ['OPENAI_API_KEY'] = key['OPENAI_API_KEY']
client = OpenAI()

# in the df data, create a new column for embeddings
data['embedding'] = data["example sentences"].apply(lambda x: get_embedding(x, model='text-embedding-3-small')).progress_apply()

# export the data to a csv file
data.to_csv("example_sentences_embeddings.csv", index=False)

