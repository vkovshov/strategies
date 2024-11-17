import os
import sys
import time
import logging
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath('../../fin_data'))

from utils.aws_conn import get_aws_client
from utils.aws_data_query import (bucket_dates, get_model_data, 
                                  ensure_s3_directory_exists, save_to_aws_bucket)

from functions import (get_accounting_variables, estimate_fair_value_ols, 
                       estimate_fair_value_ts_with_sampling_parallel, calculate_mispricing, 
                       assign_mispricing_decile, split_data_by_sector)

# Set up logging
dt_now = dt.datetime.now()
log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', dt_now.strftime("%Y-%m"))

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Configure logging
log_file = os.path.join(log_folder, f'update-{dt_now.strftime("%Y-%m-%d")}.log')
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Base paths
LOCAL_PATH = '/Users/VadimKovshov/Dropbox/INVESTMENTS/EVALUTE/STOCKS/MODEL_OUTPUTS/FUNDAMENTAL_OVER_UNDER/'
S3_PATH = 'model_output/fundamental/ts_regressor/'

# Specific paths
LOCAL_DATA_PATH = os.path.join(LOCAL_PATH, 'DATA/')
LOCAL_OUTPUT_PATH = os.path.join(LOCAL_PATH, 'OUTPUT/')
S3_DATA_PATH = os.path.join(S3_PATH, 'data/')
S3_OUTPUT_PATH = os.path.join(S3_PATH, 'output/')
S3_METRICS_PATH = os.path.join(S3_PATH, 'metrics/')

s3_bucket = 'machine-learning-evlt'
currency_reporting = 'USD'
iter_range = [10000]

def main_calcs(data=None, acc_variables=None, random_seed=42, samples=1000, sample_size=100):
    # Check if data is provided or create a sample dataset for testing purposes
    if data is None:
        np.random.seed(random_seed)

        num_firms = 1700
        num_vars = 28
        sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 
                   'Energy', 'Real Estate', 'Utilities', 'Industrials']  # Example sectors

        # Create a sample dataset with sector information
        data = pd.DataFrame({
            'compid': np.arange(num_firms),
            'ticker': [f'TKR{i}' for i in range(num_firms)],
            'market_cap': np.random.rand(num_firms) * 1000,
            'returns': np.random.randn(num_firms),
            'sector': np.random.choice(sectors, num_firms)  # Assign random sectors
        })
        
        for i in range(1, num_vars + 1):
            data[f'accounting_var{i}'] = np.random.rand(num_firms) * 100
        
        if acc_variables is None:
            acc_variables = [f'accounting_var{i}' for i in range(1, num_vars + 1)]
    
    else:
        if acc_variables is None:
            raise ValueError("acc_variables must be provided if data is provided")
        data = data.copy()

        if 'returns' not in data.columns:
            # Simulate returns column if it doesn't exist
            np.random.seed(random_seed)
            data['returns'] = np.random.randn(data.shape[0])

        if 'sector' not in data.columns:
            raise ValueError("Sector information is missing from the provided data")

    # Estimate fair values using OLS
    ols_data = estimate_fair_value_ols(data.copy(), accounting_variables=acc_variables)

    # Estimate fair values using Theil-Sen with sampling and time the process
    start_time = time.time()
    ts_data = estimate_fair_value_ts_with_sampling_parallel(data.copy(), 
                                                   accounting_variables=acc_variables, 
                                                   num_samples=samples, 
                                                   sample_size=sample_size)
    end_time = time.time()

    elapsed_time = end_time - start_time
    estimated_time_100000 = (elapsed_time / samples) * 100000

    print(f"Estimated time for 100,000 samples: {estimated_time_100000 / 60:.2f} minutes")

    # Calculate mispricing for OLS and TS
    ts_data = calculate_mispricing(ts_data, 'fair_value_ts')
    ols_data = calculate_mispricing(ols_data, 'fair_value_ols')

    # Combine OLS and TS data into a single DataFrame, including sector info
    combined_data = data[['compid', 'ticker', 'market_cap', 'sector']].copy()
    
    # Ensure 'market_cap', 'fair_value_ols', 'fair_value_ts' columns are numeric before applying formatting
    combined_data['market_cap'] = pd.to_numeric(combined_data['market_cap'], errors='coerce').fillna(0)
    ols_data['fair_value_ols'] = pd.to_numeric(ols_data['fair_value_ols'], errors='coerce').fillna(0)
    ts_data['fair_value_ts'] = pd.to_numeric(ts_data['fair_value_ts'], errors='coerce').fillna(0)

    # Format values as strings with commas
    combined_data['market_cap'] = combined_data['market_cap'].apply(lambda x: f"{x:,.0f}")
    combined_data['fair_value_ols'] = ols_data['fair_value_ols'].apply(lambda x: f"{x:,.0f}")
    combined_data['fair_value_ts'] = ts_data['fair_value_ts'].apply(lambda x: f"{x:,.0f}")

    # Ensure 'mispricing_ols_pct' and 'mispricing_ts_pct' columns are numeric and round them
    ols_data['mispricing_ols_pct'] = pd.to_numeric(ols_data['mispricing_ols_pct'], errors='coerce').fillna(0)
    ts_data['mispricing_ts_pct'] = pd.to_numeric(ts_data['mispricing_ts_pct'], errors='coerce').fillna(0)

    # Now round the numeric columns
    combined_data['mispricing_ols_pct'] = ols_data['mispricing_ols_pct'].round(2)
    combined_data['mispricing_ts_pct'] = ts_data['mispricing_ts_pct'].round(2)

    # Calculate the average mispricing percentage (optional)
    # combined_data['mispricing_avg_pct'] = ((combined_data['mispricing_ols_pct'] + combined_data['mispricing_ts_pct']) / 2).round(2)

    # Assign mispricing deciles based on the ts mispricing percentage
    combined_data = assign_mispricing_decile(combined_data, 'mispricing_ts_pct')

    # Sort the DataFrame by mispricing decile in descending order
    combined_data = combined_data.sort_values(by='mispricing_ts_decile', ascending=False)

    # print(combined_data.head())

    return combined_data

if __name__ == '__main__':
    start_time = time.time()

    s3 = get_aws_client('s3')

    for n in range (1, 10):
        # Set the data date
        model_date = bucket_dates(s3, s3_bucket_name=s3_bucket, s3_path=S3_DATA_PATH)[n]

        # Get data for regression
        extracted_data = get_model_data(s3, data_date=model_date, s3_bucket_name=s3_bucket, s3_path=S3_DATA_PATH)

        # Split the data into financials and non-financials
        financial_data, non_financial_data = split_data_by_sector(data=extracted_data)

        # Creating accounting variables list
        accounting_variables = get_accounting_variables(extracted_data)
        print(f"Accounting variables: {len(accounting_variables)}")

        # ensure_s3_directory_exists(s3, s3_bucket_name=s3_bucket, s3_output=S3_OUTPUT_PATH)

        # Combined processing for financial and non-financial data
        for data_type, data, sample_size in [("financials", financial_data, 20), ("non_financials", non_financial_data, 100)]:
            for n in iter_range:
                combined_data = main_calcs(data=data, acc_variables=accounting_variables, random_seed=42, samples=n, sample_size=sample_size)

                result_file_path = f'{S3_OUTPUT_PATH}ols_ts_mispricing_{n}_{data_type}_{currency_reporting}_{model_date.strftime("%Y%m%d")}.csv'

                save_to_aws_bucket(s3, combined_data, result_file_path, s3_bucket_name=s3_bucket, s3_output=S3_OUTPUT_PATH)
                print(f"Saved result to {result_file_path}")


        logger.info(f'Total time: {round(time.time() - start_time, 2)} seconds')
