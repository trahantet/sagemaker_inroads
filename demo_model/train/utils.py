import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np

import boto3
import pickle

from io import StringIO
import io
from gensim.models.word2vec import Word2Vec

def get_lookup_table(Key_path = 'data/Vectorized/combined_vectorized.csv'):
    s3 = boto3.client('s3')

    object = s3.get_object(Bucket='inroads-test-bucket1',Key=Key_path)
    lookup_table = pd.read_csv(io.BytesIO(object['Body'].read()), encoding='utf8', index_col=0)
    return lookup_table

def format_input(input_data):
    if (type(input_data)==list and type(input_data[0])==str):
        formatted_data=input_data
    else:
        formatted_data = input_data.encode('utf-8')
    return formatted_data
    
def lookup_table(search_table, new_input_data):
    output_list = []
    for x in new_input_data:
        #lookup_vector = np.array(search_table.loc[search_table.token==x].vector)
        lookup_vector_row = search_table.loc[search_table.token==x]
        lookup_vector = lookup_vector_row.iloc[0, 2:].values
        output_list.append(lookup_vector)
        
    return output_list