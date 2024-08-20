# Tumor Nature Prediction App

## Project Structure

.
├── api/
│   ├── api_main.py
│   ├── model.py
│   └── database.py
├── ui/
│   ├── Home.py
│   └── PastPredictions.py
├── data/
│   ├── sample_input.csv
│   └── predictions.db
├── requirements.txt
└── README.md


## Overview

This application predicts the nature of a tumor (benign or malignant) based on specific input features. Users can enter the mean radius, mean perimeter, mean texture, and mean area of the tumor through a Streamlit UI. The app supports both single-shot queries and bulk queries via CSV upload. The data is processed by a FastAPI backend, which utilizes a trained logistic regression model to make predictions. All inputs and outputs are stored in a PostgreSQL server, and past predictions can be viewed on a dedicated page within the UI.

## Features

- **Single-shot Query**: Enter tumor features manually through the UI.
- **Bulk Query**: Upload a CSV file containing multiple tumor records.
- **Prediction Storage**: Save input and output data in a PostgreSQL server.
- **Past Predictions**: View historical predictions on a dedicated UI page.

## Requirements

The project requires the following Python packages:

- `streamlit`
- `fastapi`
- `uvicorn`
- `pandas`
- `scikit-learn`
- `sqlalchemy`
- `psycopg2`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
