import pandas as pd
from loguru import logger


def cast_right_types_in_columns(df_results: pd.DataFrame, dct_col_types: dict) -> pd.DataFrame:
    """
    Cast all columns in the input dataframe with the corresponding type in the provided dictionary.

    In-place modification.

    All columns of the dataframe must have their key in the type dictionary.

    Parameters
    ----------
    df_results
        The dataframe to modify.
    dct_col_types
        The column_name -> type dictionary. The type values cna be actual typing functions or they can
        be string.

    Returns
    -------
        The dataframe with modified columns. The Dataframe is modifed in place anyway.
    """

    set_df_results_columns = set(df_results.columns)
    set_dct_col = set(dct_col_types.keys())
    diff_sets = set_df_results_columns.difference(set_dct_col)
    assert len(diff_sets) == 0, f"there are columns in df_results with no assigned type. {diff_sets}"

    for colname, str_dtype in dct_col_types.items():
        logger.debug(f"Casting: {colname}; {str_dtype}")
        try:
            if str_dtype in ["int", "float"]:
                df_results[colname] = pd.to_numeric(df_results[colname], errors="coerce")
            elif str_dtype == "datetime":
                df_results[colname] = pd.to_datetime(df_results[colname], errors="coerce")
            elif str_dtype == "bool":
                replaced_series = df_results[colname].astype(str).map({True: True, False: False, "False": False, "nan": False, "True": True})
                df_results[colname] = replaced_series
            else:
                df_results[colname] = df_results[colname].astype(str_dtype)
        except KeyError:
            logger.warning(f'{colname} is not present in dataframe.')
            df_results[colname] = None

    return df_results