import pandas as pd

def load_excel_sheets(filepath):
    return pd.read_excel(filepath, sheet_name=None)

def clean_glosses(df):
    df = df.iloc[1:, 1:]
    return df.drop(columns=['#Glosses'])
