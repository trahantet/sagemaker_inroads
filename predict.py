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

from utils import format_input, lookup_table, get_lookup_table

def model_fn(job_name = 'inroads-k-means-job-20200824001355', bucket_name = 'inroads-test-bucket1'):
    """Load the model from the `model_dir` directory."""
    print("Loading model.")
    
    # First, load the model
    model_key = job_name + '/output/model.tar.gz'
    boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
    
 
    os.system('tar -zxvf model.tar.gz')
    os.system('unzip model_algo-1')

    Kmeans_model_params = mx.ndarray.load('model_algo-1')
    #cluster_centroids = pd.DataFrame(Kmeans_model_params[0].asnumpy())
    #cluster_centroids.columns = train_df.columns
    
    
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        # model_info = torch.load(f) 
        ### how to load model info specifically for sklearn Kmeans clustering 

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    print("Done loading model.")
    return model

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
    
    output = model.predict(data)


    return result

