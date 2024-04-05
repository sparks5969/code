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


system_message = "You will judge if the given sentence is grammatically correct. ignore letter case, ignore minor issues. parsing the sentence when necessary. If correct, return 0, if incorrect, retrun 1"

# create a dataframe store the result.
grammar_check_result = pd.DataFrame(columns=["essay_id", "sentence", "incorrect"])  
issue_files = []
# loop through the rawtext folder to check grammar
# load the essay

# take two essays as an example
for file in tqdm(os.listdir("rawtext")):
    print(f'now processing file {file}')
    try:
        with open("rawtext/essay1.json" , "r") as f:
            essay = json.load(f)
            # remove the extension of the file
            essay_id = file.split(".")[0]
            for sentence in tqdm(essay):
                incorrect = check_grammar(system_message, sentence)
                new_row = pd.DataFrame({"essay_id": essay_id,
                                        "sentence": [sentence], 
                                        "incorrect": incorrect.content})
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
