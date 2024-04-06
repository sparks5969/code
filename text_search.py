"""
This script is designed to search sentences containing keywords in the essays.
I'm particularly interested in finding sentences that provide examples in the essays.
"""

from fuzzywuzzy import fuzz
import pandas as pd
import os
import json
from tqdm import tqdm

def find_sentences_with_keywords(sentences, example_keywords, threshold=80):
    example_sentences = []
    for sentence in sentences:
        for keyword in example_keywords:
            if (fuzz.partial_ratio(keyword, sentence.lower()) >= threshold) & (len(sentence.split()) > 5):
                example_sentences.append(sentence)
                break
    return example_sentences


# keywords to search for
example_keywords = ["for example", "for instance", "an example of", "illustrate", "example", "such as", "consider the case of"]

# create a dataframe to store search results. # first column: essay_id, second column: example_sentences
example_sentences_df = pd.DataFrame(columns=["essay_id", "example sentences"])

# iterate through the folder "rawtext" and search for example sentences
for filename in tqdm(os.listdir("rawtext")):
    # remove the file extension
    essay_id = filename.split(".")[0]
    # open each file and search for example sentences
    with open("rawtext/" + filename, "r") as file:
        essay = json.load(file)
        example_sentences = find_sentences_with_keywords(essay, example_keywords)
        # add the search results to the dataframe, ignore index
        df = pd.DataFrame({'essay_id': essay_id, 'example sentences': example_sentences})
        # merge df with the example_sentences_df dataframe
        example_sentences_df = pd.concat([example_sentences_df, df], ignore_index=True)

# save the search results to a csv file
example_sentences_df.to_csv("example_sentences.csv", index=False)
