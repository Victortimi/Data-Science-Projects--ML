import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor
from preprocess import scale_numeric, target, model_store
from preprocess import encode_categorical, hold_on_target, comb_scal_enco
from typing import Dict


def preprocessing_step(df: pd.DataFrame) -> pd.DataFrame:
    # scaling
    output_scale = scale_numeric(df)

    # encode cat
    output_encode = encode_categorical(df)

    # holding target in place
    output_hold = hold_on_target(df)

    # Feature engineering by combining
    preprocessed_output = comb_scal_enco(
        output_scale, output_encode, output_hold)

    return preprocessed_output


def training_data(preprocessed_df: pd.DataFrame):
    # Splitting the data
    X, y = preprocessed_df.drop([target], axis=1), preprocessed_df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training the model
    hgb_regressor = HistGradientBoostingRegressor()
    trained_model = hgb_regressor.fit(X_train, y_train)
    joblib.dump(hgb_regressor, model_store)

    return X_val, y_val, trained_model


def evaluation_trained_model(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    trained_model: HistGradientBoostingRegressor
) -> Dict[str, float]:
    # Predicting values for evaluation
    y_pred = trained_model.predict(X_val)

    # Evaluation metrics score
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))

    return {'RMSLE': rmsle}


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    # Preprocessing
    preprocessed_df = preprocessing_step(data)

    # Training
    X_val, y_val, trained_model = training_data(preprocessed_df)

    # Evaluation and result
    performance = evaluation_trained_model(X_val, y_val, trained_model)

    return performance
