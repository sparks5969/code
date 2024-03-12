"""
This python file is used to embed the text using pre-trained word embeddings.
"""
import os
import numpy as np
import openai
from openai import OpenAI
import pandas as pd

openai.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()
use_model = "text-embedding-3-small"

# Two lists of sentences
text = [
    "The cat sits outside",
    "A man is playing guitar",
    "The new movie is awesome",
]

def get_embedding(text, model=use_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model = model).data[0].embedding

df = pd.DataFrame({'combined': text})
df['my_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
