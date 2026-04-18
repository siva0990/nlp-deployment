# utils/preprocess.py

import re

def clean_text(text):
    text = str(text)

    text = re.sub(r'@[A-Za-z0-9]+', '', text)   # remove mentions
    text = re.sub(r'#', '', text)               # remove hashtags
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove links
    text = re.sub(r'[^A-Za-z\s]', '', text)     # remove special chars
    text = text.lower()                         # lowercase

    return text