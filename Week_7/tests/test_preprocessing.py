import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
from src.preprocessing import preprocess_data  

def test_preprocessing():
    data = {'Sales': [200, 300, 400, None, 500]}
    df = pd.DataFrame(data)

    test_input_path = 'data/processed/test_raw_data.csv'
    test_output_path = 'data/processed/test_cleaned_data.csv'
    df.to_csv(test_input_path, index=False)

    preprocess_data(test_input_path, test_output_path)

    processed_df = pd.read_csv(test_output_path)

    assert processed_df.isnull().sum().sum() == 0
