import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(columns=df.columns[-1], inplace=True)
    df['id'] = df['id'].astype('object')
    return df
