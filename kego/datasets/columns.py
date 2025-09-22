def assert_missing_columns(df, required_columns: list[str] | None, df_name: str = "df"):
    if required_columns is None:
        return
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{missing_columns=} in `{df_name}` with {df.columns=} but {required_columns=}"
        )
    return
