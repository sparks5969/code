"""
This script contrain functions to process pdf file, remove bibliography, split text into sentences and batches.

* function 1. convert pdf to text
* function 2. remove the bibliography from the essay
* function 3. split the text string into sentences
* function 4. split the text string into batches
"""
import re
import pyphen
import PyPDF2
import textwrap
import tiktoken
import json
from tqdm import tqdm
import nltk
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer


def pdf_to_text(path_to_file, file_name):
    """
    Fuction 1.
    This function is designed to convert PDF to text.
    :param path_to_file: The path to the PDF file.
    :param file_name: The name of the PDF file.
    :return: The text of the PDF.
    """
    # Open the PDF file
    with open(path_to_file + file_name, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages
        num_pages = len(pdf_reader.pages)

        # Extract text from each page
        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

            # Add a space between pages
            if page_num < num_pages - 1:
                text += ' '

    # Post-process the text
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)  # Insert space before capital letters
    text = re.sub(r'(\w)([^a-zA-Z0-9])', r'\1 \2', text)  # Insert space before non-alphanumeric characters
    text = re.sub(r'([^a-zA-Z0-9])(\w)', r'\1 \2', text)  # Insert space after non-alphanumeric characters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Insert space before uppercase letters following lowercase letters
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Insert space between digits and letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Insert space between letters and digits

    # Load a hyphenation dictionary
    hyphenator = pyphen.Pyphen(lang='en_US')

    # Split the text into words
    words = text.split()

    # Check each word and split if it's not a valid word
    processed_words = []
    for word in words:
        if re.fullmatch(r'[a-zA-Z]+', word):
            processed_words.append(word)
        elif re.fullmatch(r'[-+]?\d*\.?\d+', word):
            processed_words.append(word)
        else:
            # Split the word using hyphenation
            subwords = hyphenator.inserted(word).split('-')
            processed_words.extend(subwords)

    # Join the processed words back into text
    text = ' '.join(processed_words)
    pdf_file.close()
    return text


def remove_bibliography(text):
    """
    Function 2.
    This function is designed to remove the bibliographic section from the text
    :param text: the text of the pdf
    :return: the text without the bibliographic section
    """
    # Define regex patterns for common bibliographic section titles
    bibliography_patterns = [
        r'(?i)^\s*(?:References|Work Cited|Bibliography)(?:\s+(?:\w+|\S+))*\s*$',
        r'(?i)^\s*(?:References|Work Cited|Bibliography)(?:\s+(?:\w+|\S+))*\s*:',
        r'(?i)^\s*(?:References|Work Cited|Bibliography)(?:\s+(?:\w+|\S+))*\s*-',
    ]

    # Split the text into lines
    lines = text.split('\n')

    # Find the index of the bibliographic section title
    bibliography_index = None
    for i, line in enumerate(lines):
        for pattern in bibliography_patterns:
            if re.match(pattern, line):
                bibliography_index = i
                break
        if bibliography_index is not None:
            break

    # Remove the bibliographic section if found
    if bibliography_index is not None:
        text = '\n'.join(lines[:bibliography_index])

    return text


def split_text_into_sentence(text):
    """
    Function 3.
    This function is designed to split the text into sentences
    :param text: the text of the pdf
    :return: the sentences of the pdf
    """
    # Customize the tokenizer
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

    # Define a regular expression pattern for numbers with decimals and percentages
    number_pattern = r'[-+]?\d*\.?\d+%?'
    # Define a regular expression pattern for common abbreviations
    abbrev_pattern = r'\b(?:[A-Z]\.)+|\b(?:et al\.|\b(?:et al \.|e\.g\.|i\.e\.|etc\.|fig\.|vs\.)\b'


    # Modify the tokenizer's abbreviation rules
    abbrev_rules = nltk.tokenize.punkt.PunktParameters().abbrev_types
    abbrev_rules.update(set([number_pattern]))
    tokenizer._params.abbrev_types = abbrev_rules

    # Split the text into sentences
    sentences = tokenizer.tokenize(text)

    return sentences 


def split_text_into_batches(text, max_tokens=50, model='gpt2'):
    """
    Function 4
    this function is designed to split the text into batches
    :param text: the text to be split
    :param max_tokens: the maximum number of tokens per batch
    :return: a list of batches
    """
    # Load the encoding for the language model
    encoding = tiktoken.get_encoding(model)

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Initialize an empty list to store the batches
    batches = []
    
    # Initialize the current batch with an empty list
    current_batch = []
    current_batch_tokens = []
    
    for sentence in sentences:
        # Encode the sentence into tokens
        sentence_tokens = encoding.encode(sentence)
        
        # If adding the sentence to the current batch would make it too long,
        # add the current batch to the list of batches and start a new batch
        if len(current_batch_tokens) + len(sentence_tokens) > max_tokens:
            batches.append(encoding.decode(current_batch_tokens))
            current_batch = [sentence]
            current_batch_tokens = sentence_tokens
        else:
            # Otherwise, add the sentence to the current batch
            current_batch.append(sentence)
            current_batch_tokens.extend(sentence_tokens)
    
    # Add the last batch to the list of batches
    if current_batch_tokens:
        batches.append(encoding.decode(current_batch_tokens))
    
    return batches