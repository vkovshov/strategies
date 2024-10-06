from ib_insync import *
util.startLoop()
import pandas as pd
import os
from datetime import datetime

def connect_ibkr(client_id, host="127.0.0.1", port=7497):

    ib = IB()
    ib.connect(host, port, clientId=client_id)
  
    return ib

def disconnect_ibkr():

    ib = IB()
    ib.disconnect()

def fetch_positions(ib_connection, account_number):
    
    positions = ib_connection.positions(account=account_number)

    position_data = [{
        'account': item.account,
        'ticker': item.contract.symbol,
        'position': item.position,
        'average_cost': item.avgCost
    } for item in positions]

    portfolio_df = pd.DataFrame(position_data)
    
    return portfolio_df

def fetch_portfolio_value(ib_connection, account_number):
    
    # Filter the account values to get the NetLiquidationByCurrency for USD
    portfolio_value = [v for v in ib_connection.accountValues(account=account_number) 
                       if v.tag == 'NetLiquidationByCurrency' and v.currency == 'USD']

    if portfolio_value:
        return round(float(portfolio_value[0].value), 2)
    else:
        return None

def save_portfolio(portfolio_df, directory):

    current_date = datetime.now().strftime('%Y-%m-%d')

    file_name = f"momo_portfolio_{current_date}.csv"

    file_path = os.path.join(directory, file_name)

    portfolio_df.to_csv(file_path, index=False)

    return file_path

def main(client_id, account_number, save_directory):

    ib_connection = connect_ibkr(client_id)

    portfolio_df = fetch_positions(ib_connection, account_number)

    portfolio_value = fetch_portfolio_value(ib_connection, account_number)
    
    # saved_file_path = save_portfolio(portfolio_df, save_directory)

    disconnect_ibkr()

    return portfolio_df, portfolio_value

# Example usage
if __name__ == "__main__":

    client_id = 10
    account_number = "DU3208934"
    save_directory = "C:/Users/VadimKovshov/Dropbox/PYTHON/CSV_Data/momentum_model/"

    ib_connection = connect_ibkr(client_id)

    portfolio_df = fetch_positions(ib_connection, account_number)
    print(portfolio_df)

    portfolio_value = fetch_portfolio_value(ib_connection, account_number)
    print(f"Portfolio value: {portfolio_value}")

    saved_file_path = save_portfolio(portfolio_df, save_directory)
    print(f"Portfolio saved at: {saved_file_path}")

    disconnect_ibkr()
