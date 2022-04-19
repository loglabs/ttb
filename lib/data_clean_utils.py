import typing

import pandas as pd

"""
    Cleans up all strings from each column of the NYC Taxi dataset.
"""
def hard_coded_cleaning_steps_mutation(df: pd.DataFrame):
    # Correct string type confusion in vendorid.
    df.loc[df['vendorid'] == '1', 'vendorid'] = 1
    df.loc[df['vendorid'] == '1.0', 'vendorid'] = 1
    df.loc[df['vendorid'] == '2', 'vendorid'] = 2
    df.loc[df['vendorid'] == '2.0', 'vendorid'] = 2

    # Y/N more useful as 0/1 for graphing and analysis
    df.loc[df['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0
    df.loc[df['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1

    
    df.loc[df['payment_type'] == '1.0', 'payment_type'] = 1
    df.loc[df['payment_type'] == '2.0', 'payment_type'] = 2
    df.loc[df['payment_type'] == '3.0', 'payment_type'] = 3
    df.loc[df['payment_type'] == '4.0', 'payment_type'] = 4
    df.loc[df['payment_type'] == '5.0', 'payment_type'] = 5