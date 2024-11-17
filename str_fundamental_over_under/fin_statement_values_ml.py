import os
import sys
import time
import datetime as dt
from datetime import timedelta
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler

sys.path.append(os.path.abspath('../../fin_data'))

from utils.postgresql_tables import FinancialStatementLine, FinancialStatementLineAlias
from utils.helper_functions import get_test_universe_tickers, get_ticker_sector_data
from utils.date_functions import test_universe_dates
from utils.postgresql_data_query import get_company_ids
from utils.aws_data_query import ensure_s3_directory_exists, save_to_aws_bucket
from utils.aws_conn import get_aws_client

from functions import (session_scope, get_latest_balance_sheet, get_latest_4_quarters_sum, 
                       get_latest_capitalization, save_local)

# Set up logging
dt_now = dt.datetime.now()
log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', dt_now.strftime("%Y-%m"))

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Configure logging
log_file = os.path.join(log_folder, f'update-{dt_now.strftime("%Y-%m-%d")}.log')

log_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
log_handler.suffix = "%Y-%m-%d"
logging.basicConfig(
    handlers=[log_handler],
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# Define batch size and local path constants
BATCH_SIZE = 200
LOCAL_DATA_PATH = '/Users/VadimKovshov/Dropbox/INVESTMENTS/EVALUTE/STOCKS/MODEL_OUTPUTS/FUNDAMENTAL_OVER_UNDER/DATA/'
S3_DATA_PATH = 'model_output/fundamental/ts_regressor/data/'

# Define the line IDs for balance sheet, cash flow, and income statement
bs_ids = [37, 39, 40, 45, 49, 51, 52, 55, 56, 59, 60]
cf_ids = [25, 26, 27, 30, 34, 83]
is_ids = [1, 2, 3, 4, 5, 7, 10, 12, 18, 20, 23]

# Tags to be reversed for negative value differentiation
reverse_sign_tags = ['cor', 'sgna', 'rnd', 'intexp', 'taxexp', 'depamor', 
                     'payables', 'debtc', 'debtnc', 'debt']

# IDs to be excluded from division, mostly per-share data
exclude_ids = {19, 20, 23}

s3 = get_aws_client('s3')

def main_data(start_date=None, end_date=None, exclude_financial_sector=False, reverse_sign_tags=None,
              currency_reporting='USD', dimension='arq', save_to_s3=True, s3_bucket_name=None, s3_output=S3_DATA_PATH):
    """
    Processes financial statements for all companies in a 'test_universe' for each 'effective_date' 
    and saves the data either locally or to an S3 bucket for each date in the range.
    """

    if not end_date:
        end_date = dt.datetime.now().date()
    if not start_date:
        start_date = end_date

    logger.info(f"Processing test universe between {start_date} and {end_date}")

    tu_dates = test_universe_dates(start_date=start_date, end_date=end_date)
    if not tu_dates:
        logger.warning(f"No test universe dates found between {start_date} and {end_date}.")
        return None

    logger.info(f'Test universe dates: {", ".join(tu_date.strftime("%Y-%m-%d") for tu_date in tu_dates)}')

    with session_scope() as session:
        line_aliases = session.query(
            FinancialStatementLine.id,
            FinancialStatementLine.tag,
            FinancialStatementLine.name,
            FinancialStatementLineAlias.alias
        ).join(
            FinancialStatementLineAlias,
            FinancialStatementLine.id == FinancialStatementLineAlias.financial_statement_line_id
        ).all()

    line_details = {x[0]: [x[1], x[2], x[3]] for x in line_aliases}

    for tu_date in tu_dates:
        logger.info(f"\nProcessing data for {tu_date.strftime('%Y-%m-%d')}")
        start_time = time.time()

        with session_scope() as session:
            data_start = tu_date - timedelta(days=19 * 30)

            tickers = get_test_universe_tickers(session, date=tu_date, currency_reporting=currency_reporting)
            logger.info(f'Tickers in test_universe: {len(tickers)}')

            ticker_sector_data = get_ticker_sector_data(session, tickers)

            if exclude_financial_sector:
                tickers = [
                    ticker for ticker in tickers
                    if ticker_sector_data.get(ticker) and ticker_sector_data.get(ticker).strip().lower() != 'financial services'
                ]
                if not tickers:
                    logger.warning("No tickers found after applying sector exclusions.")
                    continue

            # tickers = tickers[:50] # for testing purposes

            logger.info(f'Tickers after exclusion: {len(tickers)}')

            cids_dict = get_company_ids(tickers, session, return_type='dict', batch_size=BATCH_SIZE)
            company_ids = list(cids_dict.values())

            all_errors =[]
            market_cap_values, errors = get_latest_capitalization(session, company_ids, data_start, tu_date)
            all_errors.extend(errors)
            
            balance_sheet_values, errors = get_latest_balance_sheet(session, company_ids, data_start, tu_date, dimension, bs_ids)
            all_errors.extend(errors)
            
            income_values, errors = get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, is_ids)
            all_errors.extend(errors)
            
            cash_flow_values, errors = get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, cf_ids)
            all_errors.extend(errors)
            
            if all_errors:
                logger.warning(f"Errors occurred during queries: {all_errors}")

            all_data = []

            for ticker, cid in cids_dict.items():
                data = []
                processed_line_ids = set()

                market_cap = next((val[1] for val in market_cap_values if val[0] == cid), None)
                if market_cap is not None:
                    data.append(['market_cap', market_cap])

                for val in balance_sheet_values:
                    if val[0] == cid and val[1] not in processed_line_ids:
                        _, fsl_id, date, value = val
                        if fsl_id not in exclude_ids:
                            value /= 1000000
                        data.append([line_details[fsl_id][0], value])
                        processed_line_ids.add(fsl_id)

                for val in income_values + cash_flow_values:
                    if val[0] == cid and val[1] not in processed_line_ids:
                        _, fsl_id, sum_value = val
                        if fsl_id not in exclude_ids:
                            sum_value /= 1000000
                        data.append([line_details[fsl_id][0], sum_value])
                        processed_line_ids.add(fsl_id)

                calendar_date = next((val[2] for val in balance_sheet_values if val[0] == cid), None)
                data.insert(0, ['caldate', calendar_date])
                data.insert(0, ['ticker', ticker])
                data.insert(0, ['compid', cid])

                df = pd.DataFrame(data, columns=['tag', 'value'])
                df = df.set_index('tag').transpose()

                if reverse_sign_tags:
                    cols_to_reverse = [col for col in df.columns if col in reverse_sign_tags]
                    df[cols_to_reverse] *= -1

                if df['caldate'].notnull().all():
                    all_data.append(df)

            if all_data:
                final_df = pd.concat(all_data, axis=0, ignore_index=True)

                sector_df = pd.DataFrame(list(ticker_sector_data.items()), columns=['ticker', 'sector'])
                final_df = final_df.merge(sector_df, on='ticker', how='left')

                file_name = f'aggregated_fin_statements_{tu_date.strftime("%Y%m%d")}.csv'

                if save_to_s3:
                    ensure_s3_directory_exists(s3, s3_bucket_name=s3_bucket_name, s3_output=s3_output)
                    save_to_aws_bucket(s3, final_df, file_name, s3_bucket_name=s3_bucket_name, s3_output=s3_output)
                else:
                    save_local(final_df, file_name, local_path=LOCAL_DATA_PATH)
            else:
                logger.warning(f"No data to process for date {tu_date.strftime('%Y-%m-%d')}.")

        logger.info(f"Completed processing for {tu_date.strftime('%Y-%m-%d')}.")
        logger.info(f'Total time for {tu_date.strftime("%Y-%m-%d")}: {round(time.time() - start_time, 2)} seconds')

    logger.info(f"Completed processing for all dates between {start_date} and {end_date}")

    return None


if __name__ == '__main__':
    start_time = time.time()

    main_data(
        start_date='2015-01-03',
        # end_date=dt.datetime.now().date().strftime('%Y-%m-%d'), 
        end_date='2015-01-10',
        reverse_sign_tags=reverse_sign_tags, 
        save_to_s3=True, 
        s3_bucket_name='machine-learning-evlt', 
        s3_output=S3_DATA_PATH
    )

    logger.info(f'Total time: {round(time.time() - start_time, 2)} seconds')
