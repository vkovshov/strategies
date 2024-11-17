import os
import sys
import logging
from datetime import datetime as dt, date
import pandas as pd
import numpy as np
from sqlalchemy import func, and_, select
from contextlib import contextmanager
from joblib import Parallel, delayed
from tqdm import tqdm

# Add utility paths
sys.path.append(os.path.abspath('../../fin_data'))

# Import utility modules
from utils.aws_conn import get_aws_client
from utils.aws_data_query import bucket_dates
from utils.postgresql_conn import get_session
from utils.postgresql_tables import FinancialStatementFact, CompanyDailyMetric

from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.preprocessing import StandardScaler

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define batch size and local path constants
BATCH_SIZE = 200
LOCAL_DATA_PATH = '/Users/VadimKovshov/Dropbox/INVESTMENTS/EVALUTE/STOCKS/MODEL_OUTPUTS/FUNDAMENTAL_OVER_UNDER/DATA/'

##### data extraction
@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session rollback because of exception: {e}")
        raise
    finally:
        session.close()

def get_latest_balance_sheet(session, company_ids, data_start, tu_date, dimension, bs_ids):
    """
    Fetches the latest balance sheet data for a batch of company IDs.
    
    Parameters:
    - session: SQLAlchemy session
    - company_ids: List of company IDs
    - data_start: Start date for data query
    - tu_date: Target universe date for data query
    - dimension: Data dimension (e.g., 'arq')
    - bs_ids: List of balance sheet line IDs
    
    Returns:
    - List of queried balance sheet data.
    """

    errors = []
    all_values = []
    for i in range(0, len(company_ids), BATCH_SIZE):
        batch_ids = company_ids[i:i + BATCH_SIZE]
        subquery = select(
            FinancialStatementFact.company_id,
            FinancialStatementFact.financial_statement_line_id,
            func.max(FinancialStatementFact.calendar_date).label('max_date')
        ).filter(
            and_(
                FinancialStatementFact.company_id.in_(batch_ids),
                FinancialStatementFact.financial_statement_line_id.in_(bs_ids),
                FinancialStatementFact.dimension == dimension,
                FinancialStatementFact.calendar_date >= data_start,
                FinancialStatementFact.calendar_date <= tu_date
            )
        ).group_by(
            FinancialStatementFact.company_id,
            FinancialStatementFact.financial_statement_line_id
        ).subquery()

        stmt = select(
            FinancialStatementFact.company_id,
            FinancialStatementFact.financial_statement_line_id,
            FinancialStatementFact.calendar_date,
            FinancialStatementFact.value
        ).join(
            subquery,
            and_(
                FinancialStatementFact.company_id == subquery.c.company_id,
                FinancialStatementFact.financial_statement_line_id == subquery.c.financial_statement_line_id,
                FinancialStatementFact.calendar_date == subquery.c.max_date
            )
        ).distinct()

        try:
            results = session.execute(stmt).fetchall()
            all_values.extend(results)
        except Exception as e:
            logger.error(f"Failed to execute balance sheet query: {e}")
            errors.append((batch_ids, str(e)))
            continue  # Skip this batch in case of query failure

    return all_values, errors

def get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, line_ids):
    """
    Fetches the the sum of the latest 4 quarters of data for a batch of company IDs.
    
    Parameters:
    - session: SQLAlchemy session
    - company_ids: List of company IDs
    - data_start: Start date for data query
    - tu_date: Target universe date for data query
    - dimension: Data dimension (e.g., 'arq')
    - line_ids: List of income or cash flow statement or line IDs
    
    Returns:
    - List of queried income or cash flow statement data.
    """

    errors = []
    all_values = []
    for i in range(0, len(company_ids), BATCH_SIZE):
        batch_ids = company_ids[i:i + BATCH_SIZE]
        subquery = select(
            FinancialStatementFact.company_id,
            FinancialStatementFact.financial_statement_line_id,
            FinancialStatementFact.calendar_date,
            FinancialStatementFact.value,
            func.row_number().over(
                partition_by=[FinancialStatementFact.company_id, FinancialStatementFact.financial_statement_line_id],
                order_by=[FinancialStatementFact.calendar_date.desc()]
            ).label('row_num')
        ).filter(
            and_(
                FinancialStatementFact.company_id.in_(batch_ids),
                FinancialStatementFact.financial_statement_line_id.in_(line_ids),
                FinancialStatementFact.dimension == dimension,
                FinancialStatementFact.calendar_date >= data_start,
                FinancialStatementFact.calendar_date <= tu_date
            )
        ).subquery()

        stmt = select(
            subquery.c.company_id,
            subquery.c.financial_statement_line_id,
            func.sum(subquery.c.value).label('sum_value')
        ).filter(
            subquery.c.row_num <= 4
        ).group_by(
            subquery.c.company_id,
            subquery.c.financial_statement_line_id
        )

        try:
            results = session.execute(stmt).fetchall()
            all_values.extend(results)
        except Exception as e:
            logger.error(f"Failed to execute 4 quarters sum query: {e}")
            errors.append((batch_ids, str(e)))
            continue  # Skip this batch in case of query failure

    return all_values, errors

def get_latest_capitalization(session, company_ids, data_start, tu_date):
    """
    Fetches the the latest market capitalization for a batch of company IDs.
    
    Parameters:
    - session: SQLAlchemy session
    - company_ids: List of company IDs
    - data_start: Start date for data query
    - tu_date: Target universe date for data query
    
    Returns:
    - List of queried market capitalization data.
    """
    
    errors = []
    all_values = []
    for i in range(0, len(company_ids), BATCH_SIZE):
        batch_ids = company_ids[i:i + BATCH_SIZE]
        subquery = select(
            CompanyDailyMetric.company_id,
            func.max(CompanyDailyMetric.date).label('max_date')
        ).filter(
            and_(
                CompanyDailyMetric.company_id.in_(batch_ids),
                CompanyDailyMetric.date >= data_start,
                CompanyDailyMetric.date <= tu_date
            )
        ).group_by(
            CompanyDailyMetric.company_id
        ).subquery()

        stmt = select(
            CompanyDailyMetric.company_id,
            CompanyDailyMetric.market_cap,
            CompanyDailyMetric.date
        ).join(
            subquery,
            and_(
                CompanyDailyMetric.company_id == subquery.c.company_id,
                CompanyDailyMetric.date == subquery.c.max_date
            )
        )

        try:
            results = session.execute(stmt).fetchall()
            all_values.extend(results)
        except Exception as e:
            logger.error(f"Failed to execute capitalization query: {e}")
            errors.append((batch_ids, str(e)))
            continue  # Skip this batch in case of query failure

    return all_values, errors

def save_local(final_df, file_name, local_path=LOCAL_DATA_PATH):
    """
    Saves the data locally in the specified path.
    """
    local_save_path = os.getenv('LOCAL_SAVE_PATH', local_path)
    file_path = os.path.join(local_save_path, file_name)
    
    # Save the DataFrame locally
    final_df.to_csv(file_path, index=False)
    logger.info(f'Saved locally: {file_path}')
    return file_path

##### strategy calculations
def get_accounting_variables(extracted_data, relevant_headers=None):
    """
    Generates a list of accounting variable headers from extracted data based on relevant headers.

    Parameters:
    - extracted_data (pd.DataFrame): The DataFrame containing financial data headers.
    - relevant_headers (list, optional): List of relevant headers to include. Defaults to common financial fields.

    Returns:
    - list: Filtered list of accounting variable headers found in the data.
    """
    if relevant_headers is None:
        relevant_headers = ['cashneq', 'inventory', 'receivables', 'ppnenet', 'assets', 
                            'payables', 'debtc', 'debtnc', 'debt', 'equity', 'retearn', 
                            'revenue', 'cor', 'gp', 'sgna', 'rnd', 'opinc', 'intexp', 
                            'taxexp', 'netinccmn', 'epsdil', 'dps', 'depamor', 'ncfo', 
                            'capex', 'ncfi', 'ncff', 'fcf']

    headers_list = extracted_data.columns.tolist()
    accounting_variables = [header for header in headers_list if header in relevant_headers]

    return accounting_variables

def estimate_fair_value_ols(data, accounting_variables):
    X = data[accounting_variables]
    y = data['market_cap']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    data['fair_value_ols'] = model.predict(X_scaled)
    
    return data

def fit_and_predict(sample_indices, X_all_scaled, y_all_np, model):
    """
    Fit a model on a sample and predict the entire dataset.
    """
    try:
        X_sample = X_all_scaled[sample_indices]
        y_sample = y_all_np[sample_indices]
        model.fit(X_sample, y_sample)
        return model.predict(X_all_scaled)
    except Exception as e:
        logger.error(f"Error in fit_and_predict: {e}")
        return np.full(len(y_all_np), np.nan)

def estimate_fair_value_ts_with_sampling_parallel(data, accounting_variables, num_samples=100000, sample_size=100, n_jobs=-1):
    """
    Estimates fair values using Theil-Sen regression with parallel sampling.
    """
    # Validate inputs
    if not all(var in data.columns for var in accounting_variables):
        raise ValueError("Some accounting variables are missing from the dataset.")
    if 'market_cap' not in data.columns:
        raise ValueError("The target variable 'market_cap' is missing from the dataset.")
    if sample_size > len(data):
        raise ValueError("Sample size cannot exceed the number of firms in the dataset.")

    # Prepare the predictor matrix and target variable
    X_all = data[accounting_variables]
    y_all = data['market_cap']

    # Scale the data once
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    y_all_np = y_all.values
    firms = np.arange(len(X_all_scaled))  # Numeric indices for firms

    # Parallel computation with progress tracking
    predictions_all = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(fit_and_predict)(
            np.random.choice(len(firms), size=sample_size, replace=True),
            X_all_scaled,
            y_all_np,
            TheilSenRegressor()
        ) for _ in tqdm(range(num_samples), desc="Sampling Progress")
    )

    # Aggregate predictions
    predictions_all_np = np.array(predictions_all)
    if np.isnan(predictions_all_np).any():
        logger.warning("Some predictions contain NaN values and will be excluded from aggregation.")
        predictions_all_np = predictions_all_np[~np.isnan(predictions_all_np).any(axis=1)]

    data['fair_value_ts'] = np.median(predictions_all_np, axis=0)

    return data

def evaluate_model_with_samples(data, accounting_variables, true_values, sample_sizes, sample_size=100):
    results = []
    for num_samples in sample_sizes:
        logger.info(f"Testing with num_samples={num_samples}")
        
        # Run the Theil-Sen regression with the current num_samples
        sampled_data = estimate_fair_value_ts_with_sampling_parallel(
            data, accounting_variables, num_samples=num_samples, sample_size=sample_size
        )
        
        # Calculate evaluation metrics
        mae = np.mean(np.abs(sampled_data['fair_value_ts'] - true_values))
        mape = np.mean(np.abs((sampled_data['fair_value_ts'] - true_values) / true_values)) * 100
        variance = np.var(sampled_data['fair_value_ts'])
        
        # Append results
        results.append({
            'num_samples': num_samples,
            'mae': mae,
            'mape': mape,
            'variance': variance
        })
    
    return pd.DataFrame(results)

def calculate_mispricing(data, fair_value_column):
    """
    Calculate mispricing and mispricing percentage based on fair value.
    """
    # Validate fair_value_column
    if fair_value_column not in data.columns:
        logger.warning(f"Column '{fair_value_column}' not found in DataFrame")
        raise KeyError(f"'{fair_value_column}' does not exist in the DataFrame")

    # Generate mispricing and mispricing percentage column names
    mispricing_column = f'mispricing_{fair_value_column.split("_")[-1]}'
    mispricing_pct_column = f'{mispricing_column}_pct'

    # Calculate mispricing
    data[mispricing_column] = data['market_cap'] - data[fair_value_column]

    # Set mispricing percentage to NaN by default
    data[mispricing_pct_column] = np.nan

    # Vectorized calculations for performance
    fair_value = data[fair_value_column]
    market_cap = data['market_cap']
    mispricing = data[mispricing_column]

    # Set mispricing percentage to 1000% where fair value is negative
    negative_mask = fair_value < 0
    data.loc[negative_mask, mispricing_pct_column] = 1000

    # Calculate percentage where fair value is positive and non-zero
    positive_mask = ~negative_mask & (fair_value > 0)
    data.loc[positive_mask, mispricing_pct_column] = (
        (market_cap.loc[positive_mask] / fair_value.loc[positive_mask] - 1) * 100
    )

    return data

def assign_mispricing_decile(data, mispricing_pct_column):
    data['mispricing_ts_decile'] = pd.qcut(data[mispricing_pct_column], 10, labels=False, duplicates='drop') + 1
    return data

def split_data_by_sector(data):
    """
    Splits the extracted data into two datasets: one for financial and one for non-financial companies.
    """
    # Filter financial sector
    financial_data = data[data['sector'].str.lower() == 'financial services'].reset_index(drop=True)
    
    # Filter non-financial sector
    non_financial_data = data[data['sector'].str.lower() != 'financial services'].reset_index(drop=True)

    return financial_data, non_financial_data

if __name__ == '__main__':

    s3_client = get_aws_client('s3')
    bucket = 'machine-learning-evlt'
    folder = 'model_output/fundamental/ts_regressor/data/'
    start = date(2024, 1, 1)
    end = dt.now().date()

    available_dates = bucket_dates(s3_client, s3_bucket_name=bucket, s3_path=folder, start_date=start, end_date=end)
    print("Available dates:", len(available_dates))
    print(available_dates)