# CMSC 25300/35300 Mathematical Foundations of Machine Learning
# Grad Final Project
# Satej Soman, Jonathan Tan
#
# DESCRIPTION:
# Generates features from raw data for project. Expects input data in /data/raw
# Exports matrix of numeric features to /data/final/features.csv
# Exports vector of numeric labels to data/final/labels.csv

# Setup
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

# Import data
print("Importing data")
CSV_PATH = os.path.join('..', 'data', 'raw', 'winemag-data-130k-v2.csv')
df = pd.read_csv(CSV_PATH, nrows=10000)

##############################
# PROCESS DESCRIPTION COLUMN #
##############################

print("Processing description")

# Get list of common stopwords from NLTK package
stop_words = set(stopwords.words('english'))

# convert to lowercase, remove all nonalphanumeric characters, split into
# tokens, remove stopwords
unique_words = set()
df['description'] \
    .str.lower() \
    .str.replace('[^\w\s\-]','') \
    .str.split() \
    .apply(unique_words.update)
tokens = unique_words - stop_words

# Extract matrix of one-hot encodings for description
df['description_clean'] = df['description'] \
    .str.lower() \
    .str.replace('[^\w\s\-]','') \
    .str.split()
description_enc = df \
    .apply(lambda row: [1 if token in set(row['description_clean']) else 0 for token in tokens],
           axis=1) \
    .apply(pd.Series)
description_enc.to_csv(
    os.path.join('..', 'data', 'intermediate', 'description_encoded.csv'),
    header=False,
    index=False
)

##############################
# PROCESS OTHER TEXT COLUMNS #
##############################

print("Processing other text cols")

# Test simple one-hot encoding
cols_to_enc = ['country', 'designation', 'province', 'region_1', 'region_2',
               'taster_name', 'variety', 'winery']

# Define a function to process each column
def get_one_hot_matrix(col_name):
    '''
    Takes a string column name as input, outputs a pd DataFrame containing
    one-hot encoding of the column.
    '''

    # get tokens
    col_tokens = df[col_name].unique()

    # return a matrix of one-hot encodings for each token
    col_enc = df.apply(lambda row: [1 if row[col_name] == token else 0
                                    for token in col_tokens],
                       axis=1) \
                .apply(pd.Series)

    # save intermediate file to csv, just in case
    csv_path = os.path.join('..', 'data', 'intermediate', f'{col_name}_encoded.csv')
    col_enc.to_csv(csv_path, header=False, index=False)

    return col_enc

# compute and append encoded cols
encoded_cols = [description_enc]
for i in cols_to_enc:
    encoded = get_one_hot_matrix(i)
    encoded_cols.append(encoded)

#################################
# PROCESS OTHER NUMERIC COLUMNS #
#################################

print("Processing other numeric cols")

# fill in missing values for price feature
df['price_clean'] = df['price'].fillna(df['price'].mean())
encoded_cols.append(df['price_clean'])

###############################
# EXPORT FINAL DATA FOR MODEL #
###############################

print("Exporting final data")

# compile and export final feature matrix
FINAL_DATA_PATH = os.path.join('..', 'data', 'final')
final_data = pd.concat(encoded_cols, axis=1)
final_data.to_csv(os.path.join(FINAL_DATA_PATH, 'features.csv'),
                  header=False,
                  index=False)

# export label vector
df['points'].to_csv(os.path.join(FINAL_DATA_PATH, 'labels.csv'),
                    header=False,
                    index=False)

#
