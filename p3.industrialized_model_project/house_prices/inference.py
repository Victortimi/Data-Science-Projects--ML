import numpy as np
import pandas as pd
import joblib
from preprocess import comb_scal_enco, model_store
from preprocess import model_scaler_1, model_encoder_1
from preprocess import numeric_features, categorical_features


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    scaler = joblib.load(model_scaler_1)
    scaled_data = scaler.transform(df[numeric_features])
    scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_features)
    return scaled_df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoder = joblib.load(model_encoder_1)
    encoded_data = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(data=encoded_data, columns=categorical_features)

    return encoded_df


def preprocessing_step(df: pd.DataFrame) -> pd.DataFrame:
    # scaling
    output_scale = scale_numeric(df)

    # encode cat
    output_encode = encode_categorical(df)

    # Feature engineering by combining
    preprocessed_output = comb_scal_enco(output_scale, output_encode)

    return preprocessed_output


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    # preprocessing dataframe
    preprocessed_data = preprocessing_step(input_data)

    # Loading the model for prediction
    model = joblib.load(model_store)
    prediction = model.predict(preprocessed_data)

    return prediction
