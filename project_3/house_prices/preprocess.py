import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from typing import Union, List

# Path variables
path_to_file: str = (
    'data/processed_df.parquet'
)
model_store: str = 'models/model.joblib'
model_encoder_1: str = 'models/encoder.joblib'
model_scaler_1: str = 'models/scaler.joblib'

target: str = 'SalePrice'
feature_list: List[str] = [
    'Id', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'BedroomAbvGr',
    'KitchenAbvGr', 'GarageArea', '1stFlrSF', 'MSZoning', 'Heating'
]

# Identifying and splitting features into continuous and categorical
numeric_features: List[str] = [
    'Id', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'BedroomAbvGr',
    'KitchenAbvGr', 'GarageArea', '1stFlrSF'
]

categorical_features: List[str] = ['MSZoning', 'Heating']


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df[numeric_features])
    scaled_data = scaler.transform(df[numeric_features])
    scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_features)
    joblib.dump(scaler, model_scaler_1)

    return scaled_df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OrdinalEncoder()
    encoder.fit(df[categorical_features])
    encoded_data = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(data=encoded_data, columns=categorical_features)
    joblib.dump(encoder, model_encoder_1)

    return encoded_df


# This function holds the target variable if it is trained
# data in a bid to reunite it with its features
def hold_on_target(df: pd.DataFrame) -> Union[pd.DataFrame, None]:
    if target in df.columns:
        held_target = df[target]
        held_df = pd.DataFrame(held_target)
        return held_df
    else:
        return None


# In a training data, this function recombines the scaled
# and encoded data with the target for the purpose of training
def comb_scal_enco(*dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if len(dfs) == 3:
        merged_coders = dfs[0].join(dfs[1]).join(dfs[2])
    elif len(dfs) == 2:
        merged_coders = dfs[0].join(dfs[1])
    else:
        raise ValueError("The function requires 2 or 3 DataFrames")
    return merged_coders
