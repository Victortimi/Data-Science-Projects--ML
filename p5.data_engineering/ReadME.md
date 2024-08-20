# Data Engineering Project: Client Data Warehouse and Spotify Integration

## Overview

This project focuses on the population and validation of a client data warehouse by integrating it with Spotify's database through API calls. The primary goal is to enrich the client database with Spotify's artist information, including Spotify IDs and popularity scores. Additionally, we provide an SQL script to combine artist information with genre details.

## Deliverables

1. **Data Warehouse**: A populated data warehouse containing artist information from the client database, matched with Spotify IDs and popularity scores.
2. **SQL Script**: A script that combines artist information, Spotify IDs, the number of genres each artist features in, and the number of artists in each genre.


You can install these dependencies using the following command:

```bash
pip install -r requirements.txt


.
├── data/
│   ├── client_database
│   └── spotify_database
├── notebook/
│   ├── cleaning, extraction and preparation
│   └── deliverables
├── requirements.txt
└── README.md
