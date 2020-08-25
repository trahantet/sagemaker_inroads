import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import boto3
import mxnet as mx

from __future__ import print_function
import joblib

from utils import format_input, lookup_table, get_lookup_table


def model_fn(model_dir):
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
    model = joblib.load(os.path.join(model_dir, "kmeans_cluster.joblib"))
    print("Done loading model.")

#     boto3.resource('s3').Bucket(bucket).download_file(model_key, 'model.tar.gz')
#     os.system('tar -zxvf model.tar.gz')
    
#     sagemaker_model = MXNetModel(model_data='s3://'+ model_key,
#                              role='arn:aws:iam::accid:sagemaker-role',
#                              entry_point='utils.py')
    
    
    print("Done loading model.")
    return sagemaker_model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Determining nearest cluster.')

    #calling lookuptable
    lookup_table = get_lookup_table()
    
    # process input data and turn to numpy array using lookup table
    formatted_input_data = format_input(input_data)
    vectorised_input = lookup_table(search_table = lookup_table, formatted_input_data)
    
    output = sagemaker_model.predict(vectorised_input)
    
    return result

