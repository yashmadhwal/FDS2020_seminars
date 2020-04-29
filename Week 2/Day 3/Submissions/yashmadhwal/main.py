from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
from import_1 import function_2
from import_2 import function_3
import sys
data_dir = 'data'


def flights():
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')
    jsondir = os.path.join(data_dir, 'flightjson')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join('data', 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall('data/')
        print("done", flush=True)

    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join('data', 'nycflights', '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            # Just take the first 10000 rows for the demo
            df = pd.read_csv(path).iloc[:10000] #rows about Flight Data
            df.to_json(os.path.join('data', 'flightjson', prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)

    print("** Finished! **")



def main():
    print("Setting up data directory")
    print("-------------------------")
    flights()

    print('Finished!')
    variable = sys.argv [1:]
    my_dict_variables = {}

    my_dict_variables['year'] = variable[0]
    my_dict_variables['number_of_rows'] = variable[1]
    '''
    for k,v in my_dict_variables.items():
        print(k,":",v)
    '''
    function_2(str(variable[0]),int(variable[1]))
    function_3(str(variable[0]))
    
    



if __name__ == '__main__':

    
    
    main()
    
    print('Operation Completed')