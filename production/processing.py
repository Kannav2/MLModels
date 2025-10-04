from stat import SF_FIRMLINK
import string
import pandas as pd
import numpy as np
import seaborn as sns
import toml

# decorators
from custom_logger import get_logger
from pipeline_registry import register_pipeline


@get_logger
@register_pipeline("load_data")
def load_data(file_path:str):
    suffix = file_path.split(".")[-1]

    if suffix == "csv":
        df = pd.read_csv(file_path)
    elif suffix == "xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return df
    
    

@get_logger
@register_pipeline("remove_columns")
def remove_columns(df):
    config = toml.load("pipeline.yml")
    columns_to_remove = config["columns_to_remove"]
    if columns_to_remove:
        df.drop(columns=columns_to_remove,inplace=True)
    return df