import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath('../../fin_data'))
from utils.postgresql_data_query import get_ranks

def trend_filter(df, _c, fast=64, med=128, slow=256, min_exp=0.5, med_exp=0.75, max_exp=1.0):
    # Calculate the moving averages
    ma_fast = df[_c].ewm(span=fast,min_periods=fast).mean()
    ma_med = df[_c].ewm(span=med,min_periods=med).mean()
    ma_slow = df[_c].ewm(span=slow,min_periods=slow).mean()
    px = df[_c]

    # Apply the conditional logic for 50-day moving average
    filtered_ma_fast = ma_fast.where((ma_fast >= ma_slow) & (ma_fast >= ma_med), ma_slow)
    
    # Apply the conditional logic for 100-day moving average
    filtered_ma_med = ma_med.where(ma_med >= ma_slow, ma_slow)

    # Calculate exposure based on the conditions
    conditions = [
        (px < ma_slow),
        (px >= ma_slow) & (px < filtered_ma_med),
        (px >= filtered_ma_med) & (px < filtered_ma_fast),
        (px >= filtered_ma_fast)
    ]
    
    # Define the corresponding exposure levels
    exposure_values = [0, min_exp, med_exp, max_exp]
    
    # Apply the exposure levels
    exposure = np.select(conditions, exposure_values, default=0)

    # Combine the results into a DataFrame
    trend_indicators = pd.DataFrame({
        'exposure_date': df['Date'],
        'filtered_ma_fast': filtered_ma_fast,
        'filtered_ma_med': filtered_ma_med,
        'ma_slow': ma_slow,
        'px': px,
        'exposure': exposure
    })

    return exposure # trend_indicators

def position_size_atr(account_value, risk_factor, exposure, atr):
    """
    Calculate position size using account value, risk factor, and ATR.

    Parameters:
    df (DataFrame): DataFrame containing the ATR values.
    account_value (float): Total account value.
    risk_factor (float): Percentage of account value to risk on each position.
    atr_column (str): Column name of ATR values in the DataFrame.

    Returns:
    Series: Position sizes for each stock.
    """
    position_size = (account_value * exposure * risk_factor) / atr
    return position_size

def vola_position_size(hist, max_positions=None):
    """
    Calculate position sizes using inverse volatility.

    Parameters:
    hist (DataFrame): Historical price data for the portfolio of stocks.
    max_positions (int, optional): Maximum number of positions to hold.

    Returns:
    Series: Target position weights for each stock.
    """
    # Calculate volatility for each stock (assumes daily returns are in hist)
    vola_table = hist.apply(lambda x: x.pct_change().std())

    # Calculate inverse volatility
    inv_vola_table = 1 / vola_table

    # Normalize to get target weights
    sum_inv_vola = np.sum(inv_vola_table)
    vola_target_weights = inv_vola_table / sum_inv_vola

    # If a maximum number of positions is specified, adjust the portfolio
    if max_positions is not None:
        sorted_weights = vola_target_weights.sort_values(ascending=False)
        selected_weights = sorted_weights.iloc[:max_positions]
        selected_weights /= selected_weights.sum()  # Normalize again
        return selected_weights

    return vola_target_weights

def process_universe_factors(eff_date, pr_date, lo_rank, hi_rank, model_factors, factor_cols, tickers=None):
    """
    Process universe factors for given effective and prior dates, and optionally filter by a list of tickers.
    Returns a DataFrame of the processed factor universe, optionally filtered by the given tickers.
    """
    # Get universe factor ranks for both dates
    ranks_eff = get_ranks([eff_date], lo_rank, hi_rank, factors=model_factors).reset_index()
    ranks_pr = get_ranks([pr_date], lo_rank, hi_rank, factors=model_factors).reset_index()

    # Merge the ranks from both dates
    ranks = pd.merge(ranks_eff, ranks_pr, on=['ticker', 'factor_definition_id'], 
                     suffixes=('_eff', '_pr'), how='outer')
    
    # Create a new 'rank' and 'factor_date' columns that prioritize eff_date data
    ranks['rank'] = ranks['rank_eff'].combine_first(ranks['rank_pr'])
    ranks['factor_date'] = ranks['factor_date_eff'].combine_first(ranks['factor_date_pr'])
    
    # Filter out rows where ranks came from different dates to ensure consistency
    ranks = ranks.dropna(subset=['rank_eff'])\
                 .drop(['rank_pr', 'rank_eff', 'factor_date_pr', 'factor_date_eff'], axis=1)\
                 .reset_index(drop=True)[factor_cols]

    # Filter by the provided list of tickers if 'tickers' is not None
    if tickers is not None:
        ranks = ranks[ranks['ticker'].isin(tickers)]

    # Pivot the DataFrame so that each ticker has a single row with factors as columns
    ranks_pivoted = ranks.pivot_table(index=['ticker', 'factor_date'], 
                                      columns='factor_definition_id', 
                                      values='rank')
    
    # Rename columns for clarity, sort by ticker, date and reindex
    ranks_pivoted.columns.name = None
    ranks_pivoted.columns = [f'f_{col}' if isinstance(col, int) else col for col in ranks_pivoted.columns]
    ranks = ranks_pivoted.sort_values(by=['ticker', 'factor_date']).reset_index(drop=False)
    
    print(f'Test universe: {len(ranks)}')
    return ranks


def signal_filter(df, _h, _l, _c, n, pct_change_threshold=0.15, period=90):
    """
    Generate a trading signal based on the closing price and an exponential moving average (EMA),
    with an additional check for whether the stock moved more than a certain percentage
    up or down in the last `period` days.
    Returns:
    - signal_value: 1 if the last close > EMA and no significant move detected in the last `period` days, otherwise 0
    """

    # Calculate the Exponential Moving Average (EMA)
    df['EMA'] = df[_c].ewm(span=n, min_periods=n).mean()

    # Detect big moves (either low to previous high or high to previous low exceeding the threshold)
    df['big_move'] = (
        ((df[_l] / df[_h].shift(1) - 1).abs() >= pct_change_threshold) | 
        ((df[_h] / df[_l].shift(1) - 1).abs() >= pct_change_threshold)
    ).astype(int)

    # Find if there's been a big move in the last `period` days
    last_period_big_move = df['big_move'].iloc[-period:].max()

    # Generate the signal: 1 if last close > EMA and no big move in the last `period` days, otherwise 0
    signal_value = 0
    if not df.empty and not df['EMA'].isna().iloc[-1]:
        last_close = df[_c].iloc[-1]
        last_ema = df['EMA'].iloc[-1]

        signal_value = 1 if (last_close > last_ema) and (last_period_big_move == 0) else 0

    return signal_value

def select_positions_1(df, max_exposure=1, max_sector_allocation=0.2):
    """
    Prioritize existing positions, then new positions with activation, then new positions, 
    and finally old positions. If a position is not selected, the position size is set to 0,
    but the rest of the information is retained.
    """
    selected_positions = []
    total_exposure = 0.0
    sector_allocations = {}
    selected_tickers = set()  # Keep track of selected tickers

    def try_select_position(row):
        nonlocal total_exposure, sector_allocations
        position_pct = row['pct_position']
        sector = row['sector']
        
        # Calculate the potential new sector allocation
        new_sector_allocation = sector_allocations.get(sector, 0) + position_pct
        
        # Check if adding this position would exceed the sector allocation limit or total exposure
        if new_sector_allocation <= max_sector_allocation and (total_exposure + position_pct) <= max_exposure:
            selected_positions.append(row)
            selected_tickers.add(row['ticker'])  # Add the ticker to the selected set
            total_exposure += position_pct
            sector_allocations[sector] = new_sector_allocation
            
            # Stop if we've reached the maximum allowed exposure
            return total_exposure >= max_exposure
        return False

    # First pass: prioritize existing positions
    for index, row in df[(df['label'] == 'position') & (df['activation_value'] != 1)].iterrows():
        if try_select_position(row):
            break

    # Second pass: prioritize new positions with activation
    for index, row in df[(df['label'] == 'position') & (df['activation_value'] == 1)].iterrows():
        if try_select_position(row):
            break

    # Third pass: handle remaining new positions without activation
    for index, row in df[(df['label'] == 'new') & (df['activation_value'] != 1)].iterrows():
        if try_select_position(row):
            break

    # Fourth pass: handle remaining new positions with activation
    for index, row in df[(df['label'] == 'new') & (df['activation_value'] == 1)].iterrows():
        if try_select_position(row):
            break

    # Fifth pass: handle remaining positions (neither new nor existing)
    for index, row in df.iterrows():
        if row['label'] not in ['position', 'new']:
            if try_select_position(row):
                break

    # Convert the selected positions list back to a DataFrame
    selected_df = pd.DataFrame(selected_positions)

    # For positions that were not selected, set their position size to 0 but keep the rest of the information
    unselected_df = df[~df['ticker'].isin(selected_tickers)].copy()
    
    # Set position size to 0 for unselected positions
    unselected_df['pct_position'] = 0
    unselected_df['dollar_position'] = 0  
    unselected_df['new_position'] = 0  

    # Concatenate the selected and unselected DataFrames
    final_df = pd.concat([selected_df, unselected_df], ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df

def select_positions_2(df, max_exposure=1, max_sector_allocation=0.2): # priority is given to 'new' positions
    selected_positions = []
    total_exposure = 0.0
    sector_allocations = {}

    def try_select_position(row):
        nonlocal total_exposure, sector_allocations
        position_pct = row['pct_position']
        sector = row['sector']
        
        # Calculate the potential new sector allocation
        new_sector_allocation = sector_allocations.get(sector, 0) + position_pct
        
        # Check if adding this position would exceed the sector allocation limit or total exposure
        if new_sector_allocation <= max_sector_allocation and (total_exposure + position_pct) <= max_exposure:
            selected_positions.append(row)
            total_exposure += position_pct
            sector_allocations[sector] = new_sector_allocation
            
            # Stop if we've reached the maximum allowed exposure
            return total_exposure >= max_exposure
        return False

    # First pass: prioritize 'new' positions
    for index, row in df[df['label'] == 'new'].iterrows():
        if try_select_position(row):
            break

    # Second pass: go through the entire DataFrame (which will include both 'new' and 'old' positions)
    for index, row in df.iterrows():
        if row['label'] != 'new':  # Skip 'new' positions that have already been considered
            if try_select_position(row):
                break

    # Convert the selected positions list back to a DataFrame
    selected_df = pd.DataFrame(selected_positions)
    return selected_df

def exposure_summary(df, group_keys=['sector', 'industry', 'ticker', 'name'], 
                     aggregate_by='exposure', 
                     round_val=2, format_str='{:.02f}'):
    
    # Group by the specified keys and calculate the sum for the exposure
    summary_df = df.groupby(group_keys)['pct_position'].sum().reset_index()
    
    # Rename the aggregated column to 'exposure'
    summary_df.rename(columns={'pct_position': aggregate_by}, inplace=True)
    
    # Additional aggregation for sector and industry levels
    sector_agg = summary_df.groupby('sector')[aggregate_by].sum().reset_index().rename(columns={aggregate_by: 'sector_exp'})
    industry_agg = summary_df.groupby(['sector', 'industry'])[aggregate_by].sum().reset_index().rename(columns={aggregate_by: 'industry_exp'})
    
    # Merge the sector and industry exposures back to the summary DataFrame
    summary_df = summary_df.merge(sector_agg, on='sector')
    summary_df = summary_df.merge(industry_agg, on=['sector', 'industry'])
    
    # Sort by sector_exp, industry_exp, and exposure
    summary_df = summary_df.sort_values(by=['sector_exp', 'sector', 'industry_exp', 'industry', aggregate_by, 'ticker', 'name'], 
                                        ascending=[False, True, False, True, False, True, True]).round(round_val)
    
    # Reorder columns to have the name column at the end
    ordered_columns = ['sector', 'sector_exp', 'industry', 'industry_exp', 'ticker', aggregate_by, 'name']
    summary_df = summary_df[ordered_columns]
    
    # Reset index and calculate the total exposure
    summary_df = summary_df.reset_index(drop=True)
    total_exposure = round(summary_df[aggregate_by].sum(), round_val)
    print(f'Total exposure: {total_exposure}')
    
    # Apply the background gradient to the exposure columns and format them
    exposure_cols = ['sector_exp', 'industry_exp', aggregate_by]
    summary_df_styled = summary_df.style.background_gradient(subset=exposure_cols, cmap='RdYlGn').format({col: format_str for col in exposure_cols})
    
    return summary_df_styled

def filter_ranks(df, filter=1):

    filter = 'filter_' + str(filter)

    if filter == 'filter_1':
        filter_1 = df[(df['f_85'] > 85) & (df['f_86'] >= 80) & (df['f_85'] >= df['f_86'])].copy()
        filter_1 = filter_1.sort_values(by=['f_85', 'f_86'], ascending=False)
        return filter_1
    
    elif filter == 'filter_2':
        filter_2 = df[(df['f_85'] > 80) & (df['f_86'] >= 70)].copy()
        filter_2 = filter_2.sort_values(by=['f_85', 'f_86'], ascending=False)
        return filter_2
    
    elif filter == 'filter_3':
        filter_3 = df[(df['f_86'] > 90) & (df['f_85'] > 80)].copy()
        filter_3 = filter_3.sort_values(by=['f_86', 'f_85'], ascending=False)
        return filter_3
    
    if filter == 'filter_4':
        filter_4 = df[(df['f_85'] > 85) & (df['f_86'] >= 80) ].copy()
        filter_4 = filter_4.sort_values(by=['f_85', 'f_86'], ascending=False)
        return filter_4
    
    else:
        return print('No filter is selected...')

def read_prior_activations(model_output):
    # Assuming the CSV contains columns 'ticker' and 'activation_prior'
    prior_activations = pd.read_csv(model_output)
    return prior_activations[['ticker', 'activation_value']]

def step_activation(signal_now, signal_prior, activation_value, label, extreme_threshold=95, activation_mult=2):

    if signal_prior < extreme_threshold and signal_now >= extreme_threshold and label == 'position':
        return activation_mult
    
    elif activation_value != 1 and label == 'position':
        return activation_value

    return 1

def construct_trade_df(positions_df, new_positions_df):
    """
    Construct a DataFrame with buy and sell orders by comparing existing positions and selected positions.
    Returns a DataFrame identifying necessary trades, sorted by sector dollar_position and ticker.
    """

    # Merge existing positions with selected positions on 'ticker'
    trade_df = pd.merge(positions_df, new_positions_df, on='ticker', how='outer', suffixes=('_existing', ''))

    # Calculate shares to trade (if position exists in both DataFrames)
    trade_df['shares_to_trade'] = trade_df['new_position'].fillna(0) - trade_df['position'].fillna(0)

    # Fill missing values for positions that don't exist in either of the DataFrames
    trade_df['position'].fillna(0, inplace=True)  # No position in existing portfolio
    trade_df['new_position'].fillna(0, inplace=True)  # No new position in the selected portfolio

    # Sort by sector aggregated dollar_position and then ticker
    trade_df['sector_dollar_position'] = trade_df.groupby('sector')['dollar_position'].transform('sum')
    trade_df.sort_values(by=['sector_dollar_position', 'ticker'], ascending=[False, True], inplace=True)

    # Select the relevant columns for the final trade DataFrame
    trade_df = trade_df[['ticker', 'label', 'position', 'shares_to_trade', 'new_position', 'average_cost', 
                         'dollar_position', 'pct_position', 'ewma_stop', 'name', 'sector']]

    # Filter out records where both position == 0 and shares_to_trade == 0
    trade_df = trade_df[(trade_df['position'] != 0) | (trade_df['shares_to_trade'] != 0)]
    trade_df.reset_index(drop=True, inplace=True)

    return trade_df