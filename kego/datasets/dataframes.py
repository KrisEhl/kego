def retain_df(df, reset_index=True):
    df = df.copy()
    if reset_index:
        df = df.reset_index()
    return df
