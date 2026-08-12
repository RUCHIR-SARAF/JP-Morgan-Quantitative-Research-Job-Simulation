# J.P. Morgan Quantitative Research Job Simulation

## Overview

This repository contains Python-based financial analysis and problem-solving work completed as part of a J.P. Morgan Quantitative Research Job Simulation

## Repository Structure

```text
JP-Morgan-Quantitative-Research-Job-Simulation/
│
├── README.md
|
├── data/
|
├── scripts/
|
├── notebooks/
│   ├── Bucket FICO scores.ipynb
│   ├── contract_price_estimator.ipynb
│   ├── Credit_risk_analysis.ipynb
│   └── Natural_gas.ipynb
│
├── .gitignore
|
└── J.P. Morgan Certificate

```

## Projects

### 1. Credit Risk Analysis
**Notebook:** `credit_risk_analysis.ipynb`

Analysis of a 10,000-borrower loan dataset containing credit lines, loan amount, total debt, income, years employed, FICO score, and default status.

Topics covered:
- Exploratory data analysis and descriptive statistics
- Default-rate and correlation analysis
- Train/test split and feature scaling
- Logistic Regression
- Random Forest Classification
- Probability of Default (PD) estimation
- AUC, Log Loss, and Brier Score
- Expected Loss using `EL = PD × LGD × EAD`

### 2. FICO Score Bucketing
**Notebook:** `Bucket_FICO_scores.ipynb`

Develops an approach for converting continuous FICO scores into discrete risk ratings.

Topics covered:
- FICO score aggregation
- Default-rate analysis
- Bucket log-likelihood
- Prefix-sum calculations
- Dynamic programming for optimal bucket boundaries
- FICO rating assignment
- Default-rate analysis by rating

The notebook creates 10 FICO-based rating buckets and evaluates the observed default rate across them.

### 3. Natural Gas Price Analysis
**Notebook:** `Natural_Gas.ipynb`

Works with monthly natural gas price data.

Topics covered:
- Data loading and inspection
- Date conversion and sorting
- Time-series data preparation
- Historical price analysis and visualization

The dataset contains 48 monthly observations from October 2020 through September 2024.

### 4. Natural Gas Contract Price Estimator
**Notebook:** `contract_price_estimator.ipynb`

Develops a reusable `contract_price_estimation()` function for estimating the value of a natural gas storage contract.

The function considers:
- Injection and withdrawal schedules
- Injection and withdrawal rates
- Maximum storage capacity
- Storage costs
- Variable injection/withdrawal costs
- Natural gas prices

It validates operational constraints and calculates:
- Sales revenue
- Purchase cost
- Storage cost
- Variable fees
- Ending inventory
- Net contract value

A sample calculation in the notebook produces a contract value of **770.00** under the provided assumptions.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook



## Key Skills Demonstrated

- Financial data analysis
- Exploratory data analysis
- Credit risk modelling
- Probability of Default estimation
- FICO risk segmentation
- Classification modelling
- Model evaluation
- Expected Loss calculation
- Time-series data preparation
- Financial contract valuation
- Inventory and capacity constraint handling
- Python function development
- Data manipulation with Pandas and NumPy

## Certificate

This repository accompanies the J.P. Morgan Quantitative Research Job Simulation coursework and contains the Python-based financial analysis work completed during the program.

## Disclaimer

The notebooks are provided for educational and portfolio purposes and uses the datasets and assumptions included in the coursework.
