import os
import json
from tqdm import tqdm

from nlp_methods import pdf_to_text
from nlp_methods import remove_bibliography
from nlp_methods import split_text_into_sentence
from nlp_methods import split_text_into_batches
import wordsegment
from wordsegment import load, segment
load()

# specify the local path to the pdf files
folder_path = '/Users/sining/Library/CloudStorage/GoogleDrive-sxw924@case.edu/My Drive/Research/pedagogical research/GenerativeAI for teaching and learning/GenAI in Essay Writing/NudgeEssayResearch/clean_file/S23/'
# get a list of the pdf files in the directory
pdf_files = os.listdir(folder_path)


# process the pdf files automatically. If the file not found, catch the exception, skip the file and print the error message
error_file = []
for file in tqdm(pdf_files):
    print(f'Processing {file}')
    try:
        text = pdf_to_text(folder_path, file)  # convert pdf into text string
        text = remove_bibliography(text)  # remove bibliography from the text
        sentences = split_text_into_sentence(text)  # split the text into sentences
        sentences = [' '.join(segment(sentence)) for sentence in sentences]   # segment the words in each sentence     
        # remove the .pdf from the file name
        file_name = file.replace('.pdf', '')
        # export the batches to a JSON file, use the file name as the key, and the batches as the value
        json.dump(sentences, open('rawtext/' + file_name + '.json', 'w'))
        print(f'Successfully Processed {file}')
    except Exception as e:
        print(f'Error processing {file}: {e}')
        error_file.append(file)


# export error file as txt
with open('error_file.txt', 'w') as f:
    for item in error_file:
        f.write("%s\n" % item)
    
print("All DONE!====================================")

