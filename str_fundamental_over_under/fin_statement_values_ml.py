import sys
import os
sys.path.append(os.path.abspath('../../fin_data'))
from utils.postgresql_conn import get_session
from utils.postgresql_tables import (FinancialStatementLine, FinancialStatementLineAlias, 
                                     FinancialStatementFact, CompanyDailyMetric)
from utils.helper_functions import get_test_universe_tickers, get_ticker_sector_data
from utils.date_functions import test_universe_dates
from utils.postgresql_data_query import get_company_ids
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import func, and_, select
import time
import logging
import boto3
import io
from contextlib import contextmanager
from botocore.exceptions import BotoCoreError, ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the line IDs for balance sheet, cash flow, and income statement
bs_ids = [37, 39, 40, 45, 49, 51, 52, 55, 56, 59, 60]
cf_ids = [25, 26, 27, 30, 34, 83]
is_ids = [1, 2, 3, 4, 5, 7, 10, 12, 18, 20, 23]

reverse_sign_tags = ['cor', 'sgna', 'rnd', 'intexp', 'taxexp', 'depamor', 
                     'payables', 'debtc', 'debtnc', 'debt']  # tags to be reversed

exclude_ids = {19, 20, 23}  # IDs to be excluded from division

BATCH_SIZE = 200  # Adjust the batch size as needed

# Context manager for session handling
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
            continue  # Skip this batch in case of query failure

    return all_values


def get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, line_ids):
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
            continue  # Skip this batch in case of query failure

    return all_values


def get_latest_capitalization(session, company_ids, data_start, tu_date):
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
            continue  # Skip this batch in case of query failure

    return all_values


def ensure_s3_directory_exists(s3_client, bucket, prefix):
    """
    Ensures the directory (prefix) exists in S3 by checking if any objects exist with the given prefix.
    If the prefix does not exist, create a placeholder file to make the directory.
    """
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    # If no objects exist in this prefix, we create a placeholder to make the directory
    if 'Contents' not in response:
        placeholder_key = f"{prefix}placeholder.txt"
        s3_client.put_object(Bucket=bucket, Key=placeholder_key, Body='This is a placeholder to create the directory.')
        logger.info(f'Created S3 directory with placeholder: {bucket}/{placeholder_key}')
    else:
        logger.info(f'S3 directory exists: {bucket}/{prefix}')


def save_local(final_df, file_name):
    """
    Saves the data locally in the specified path.
    """
    local_save_path = os.getenv('LOCAL_SAVE_PATH', '/Users/VadimKovshov/Dropbox/INVESTMENTS/EVALUTE/STOCKS/MODEL_OUTPUTS/FUNDAMENTAL_OVER_UNDER/DATA/')
    file_path = os.path.join(local_save_path, file_name)
    
    # Save the DataFrame locally
    final_df.to_csv(file_path, index=False)
    logger.info(f'Saved locally: {file_path}')
    return file_path


def main(start_date=None, end_date=None, exclude_financial_sector=False, reverse_sign_tags=None,
         currency_reporting='USD', dimension='arq', save_to_s3=False, s3_bucket_name=None, s3_output=None):
    """
    Processes financial statements for all companies in a 'test_universe' for each 'effective_date' 
    and saves the data either locally or to an S3 bucket for each date in the range.
    """

    if not end_date:
        end_date = datetime.now().date()
    if not start_date:
        start_date = end_date

    logger.info(f"Processing test universe between {start_date} and {end_date}")

    tu_dates = test_universe_dates(start_date=start_date, end_date=end_date)
    if not tu_dates:
        logger.warning(f"No test universe dates found between {start_date} and {end_date}.")
        return None

    logger.info(f'Test universe dates: {", ".join(tu_date.strftime("%Y-%m-%d") for tu_date in tu_dates)}')

    # Move line_aliases query outside the loop if it doesn't depend on tu_date
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
        logger.info(f"Processing data for {tu_date.strftime('%Y-%m-%d')}")
        start_time = time.time()  # Start timer for this tu_date

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

            # tickers = tickers[:50] # Limit the number of tickers for testing

            logger.info(f'Tickers after exclusion: {len(tickers)}')

            cids_dict = get_company_ids(tickers, session, return_type='dict', batch_size=BATCH_SIZE)
            company_ids = list(cids_dict.values())

            market_cap_values = get_latest_capitalization(session, company_ids, data_start, tu_date)
            balance_sheet_values = get_latest_balance_sheet(session, company_ids, data_start, tu_date, dimension, bs_ids)
            income_values = get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, is_ids)
            cash_flow_values = get_latest_4_quarters_sum(session, company_ids, data_start, tu_date, dimension, cf_ids)

            all_data = []  # Reset all_data for each tu_date

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
                    # Reverse the sign only for the columns present in df
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
                    s3_output_path = s3_output if s3_output else 'machine_learning_evlt/model_output/fundamental/ts_regressor/data/'

                    s3 = boto3.client('s3', region_name='eu-central-1')
                    ensure_s3_directory_exists(s3, s3_bucket_name, s3_output_path)
                    s3_data_path = os.path.join(s3_output_path, file_name)

                    csv_buffer = io.StringIO()
                    final_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    try:
                        s3.put_object(Bucket=s3_bucket_name, Key=s3_data_path, Body=csv_buffer.getvalue())
                        logger.info(f'File uploaded to S3: {s3_bucket_name}/{s3_data_path}')
                    except (BotoCoreError, ClientError) as e:
                        logger.error(f"Failed to upload to S3: {e}")
                else:
                    save_local(final_df, file_name)
            else:
                logger.warning(f"No data to process for date {tu_date.strftime('%Y-%m-%d')}.")

        logger.info(f"Completed processing for {tu_date.strftime('%Y-%m-%d')}.")
        logger.info(f'Total time for {tu_date.strftime("%Y-%m-%d")}: {round(time.time() - start_time, 2)} seconds')

    logger.info(f"Completed processing for all dates between {start_date} and {end_date}")

    return None


if __name__ == '__main__':
    start_time = time.time()

    extract = main(start_date='2023-02-01', end_date='2023-02-08', reverse_sign_tags=reverse_sign_tags, 
                   save_to_s3=False, s3_bucket_name='machine-learning-evlt', s3_output=None)
    
    logger.info(f'Total time: {round(time.time() - start_time, 2)} seconds')