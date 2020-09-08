from __future__ import print_function
import argparse
import json
import os
import pickle
import sys
#import sagemaker_containers
import pandas as pd
import numpy as np
import boto3
#import mxnet as mx

import joblib

from utils import format_input, lookup_table, get_lookup_table

## Import any additional libraries you need to define a model
from sklearn.cluster import KMeans

    
# Provided model load function
def model_fn(model_dir = 'model/model.tar.gz'):
    """Load the Sklearn model from the `model_dir` directory."""
    print("Loading model.")

#     # First, load the parameters used to create the model.
#     job_name = 'inroads-k-means-job-20200824001355'
#     model_key = "kmeans/" + job_name + "/output/model.tar.gz"
    
    # Ioana: 25.08.2020
    # load using joblib
    # You'll note that this is significantly shorter to write out. 
    # We will probably have to upgrade this to be able to take from an s3 bucket
    # but you've already done so below
#     model = joblib.load(os.path.join(model_dir, "kmeans_cluster.joblib"))
#     print("Done loading model.")

    boto3.resource('s3').download_file(bucket_name, model_dir, 'model.tar.gz')
    sagemaker_model=os.system('tar -zxvf model.tgz')
        
    
    print("Done loading model.")
    return 


def input_fn(serialized_input_data, content_type ='text/csv'):
    print('Deserializing the input data.')
    if content_type == 'text/csv':
        try: 
            data = serialized_input_data.decode('utf-8')
        except:
            data = serialized_input_data
        print(data)
        print(type(data))
        
        #calling lookuptable
        vector_table = get_lookup_table()

        # process input data and turn to numpy array using lookup table
        formatted_input_data = format_input(data)
        
        print(formatted_input_data)
        print(type(formatted_input_data))
        
        vectorised_input = lookup_table(vector_table, formatted_input_data)
        
        return vectorised_input
    
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Determining nearest cluster.')


    
    output = model.predict(input_data)
    
    return output

