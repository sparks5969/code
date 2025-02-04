"""
This script is designed to check grammar in submitted essays
there is no need to correction. If a sentence is grammatically incorrect, mark it as incorrect.
"""

import json
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from openai import OpenAI
import os
import boto3
from get_secret import get_secret

key = get_secret()
key = json.loads(key)
os.environ['OPENAI_API_KEY'] = key['OPENAI_API_KEY']
client = OpenAI()

def check_grammar(system_message, sentence):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1
    )
    return response.choices[0].message


system_message = """You will rate the quality of given sentences.
Please rate the quality of the following sentences on continuous scale of 0 to 1.
If a sentence is perfectly written, please rate it as 1.
if a sentence is poorly written, please rate it as 0.
if a sentence is somewhere in between, please rate it as a decimal between 0 and 1.
ignore letter case.
parsing the sentence when necessary. """

# create a dataframe store the result.
grammar_check_result = pd.DataFrame(columns=["essay_id", "sentence", "quality_rating"])  
issue_files = []
# loop through the rawtext folder to check grammar
# load the essay

file_list = os.listdir("rawtext")

for file in tqdm(file_list):
    print(f'now processing file {file}')
    try:
        with open("rawtext/" + file , "r") as f:
            essay = json.load(f)
            # remove the extension of the file
            essay_id = file.split(".")[0]
            for sentence in tqdm(essay):
                # if the sentence is too short, skip it
                if len(sentence.split()) < 5:
                    continue
                quality_rating = check_grammar(system_message, sentence)
                new_row = pd.DataFrame({"essay_id": essay_id,
                                        "sentence": [sentence], 
                                        "quality_rating": quality_rating.content})
                grammar_check_result = pd.concat([grammar_check_result, new_row], ignore_index=True)
    except Exception as e:
        print(f"Error processing file {file}")
        issue_files.append(file)
        continue

# export the result to a csv file
grammar_check_result.to_csv("grammar_check_result.csv", index=False)  

# export the issue file to a txt file
with open("issue_files.txt", "w") as f:
    for file in issue_files:
        f.write(file)
        f.write("\n")
