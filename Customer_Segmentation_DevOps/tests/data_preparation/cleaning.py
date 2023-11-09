import unittest
import pandas as pd
import json  
import os
import sys

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the src directory
src_dir = os.path.join(script_dir, '..', '..', 'src')

# Add the src directory to the module search path
sys.path.append(src_dir)
from data_preparation.cleaning import prepare_data, drop_c_id, clean_data

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file for testing
        self.test_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'order_status': ['delivered', 'shipped', 'delivered']
            # Add other necessary columns for your tests
        })
        self.temp_file = 'test_data.csv'
        self.test_data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        # Remove the temporary CSV file
        os.remove(self.temp_file)

    def test_prepare_data(self):
        df = prepare_data(self.temp_file)
        # Add your assertions based on the expected behavior of prepare_data
        self.assertTrue('customer_id' in df.columns)
        self.assertTrue('order_status' in df.columns)
        # Add more assertions as needed

    def test_drop_c_id(self):
        df = drop_c_id(self.test_data)
        # Add your assertions based on the expected behavior of drop_c_id
        self.assertTrue('customer_id' in df.columns)
        self.assertFalse('customer_unique_id' in df.columns)
        # Add more assertions as needed

    def test_clean_data(self):
        df = clean_data(self.test_data)
        # Add your assertions based on the expected behavior of clean_data
        self.assertTrue('customer_id' in df.columns)
        self.assertTrue('order_status' in df.columns)
        self.assertEqual(df['order_status'].unique(), ['delivered'])
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
