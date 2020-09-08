from __future__ import print_function

import argparse
import os
import pandas as pd
import joblib
import json
import pickle
import sys
import numpy as np
import boto3

## TODO: Import any additional libraries you need to define a model
from sklearn.cluster import KMeans
from utils import format_input, lookup_table, get_lookup_table #, upload_lookup_table

#Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    #parser.add_argument('--n-clusters', type=int, default=3)

    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "inroads_data_test.csv"), header=0, names=None, index_col=0)
        

    ## TODO: Define a model 
    model = KMeans(
            n_clusters = 9,
            random_state=100,
            init='k-means++'
            #n_clusters=args.n_clusters,
            )
    
    
    ## TODO: Train the model
    model.fit(train_data)
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
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
        vectorised_input,vector_table_new = lookup_table(vector_table, formatted_input_data)
        
        #upload_lookup_table()
        
        return vectorised_input
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Determining nearest cluster.')

    output = model.predict(input_data)
    
    return output