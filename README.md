# Airline Passenger Satisfaction Prediction

**Module:** 6001CMD – Machine Learning  
**Project:** Group Assignment  
**Problem Category:** Binary Classification  
**Programming Language:** Python 3.12  
**Environment:** Jupyter Notebook

## Project Overview

This project develops and evaluates machine learning models to predict airline passenger satisfaction using passenger demographics, travel information, and service ratings.

The objective is to help airlines identify potentially dissatisfied passengers and support data-driven service improvements.

## Dataset

**Dataset:** Airline Passenger Satisfaction  
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

- Training records: 103,904
- Testing records: 25,976
- Target: Passenger satisfaction
- Problem type: Binary classification

Download `train.csv` and `test.csv` from Kaggle and place them in a folder named `data` before running the notebook.

## Machine Learning Models

Three individual classification models were developed:

1. Logistic Regression
2. Decision Tree
3. Random Forest

A Weighted Soft Voting ensemble combining Decision Tree and Random Forest was also evaluated.

## Methodology

The project includes:

1. Exploratory Data Analysis
2. Data Preprocessing and Feature Engineering
3. Base Model Development
4. Hyperparameter Optimisation
5. Ensemble Learning
6. Model Evaluation and Comparison

## Final Test Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 87.1% | 0.869 |
| Decision Tree | 95.4% | 0.953 |
| Random Forest | 96.3% | 0.962 |
| Weighted Soft Voting | 96.0% | 0.960 |

**Recommended Model:** Random Forest

Random Forest achieved the strongest overall predictive performance and identified dissatisfied passengers more effectively than the ensemble.

## Project Files

- `Airline_Passenger_Satisfaction.ipynb` – Complete Jupyter Notebook
- `Airline_Passenger_Satisfaction.py` – Python source code

## Requirements

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- SciPy

## Running the Project

1. Download the project files.
2. Download the dataset from the Kaggle link above.
3. Create a `data` folder containing `train.csv` and `test.csv`.
4. Open `Airline_Passenger_Satisfaction.ipynb` in Jupyter Notebook.
5. Select **Restart Kernel and Run All Cells**.

The notebook performs preprocessing, trains and evaluates the models, and generates the results and visualisations.

## Academic Purpose

This repository contains source code developed for the 6001CMD Machine Learning group assignment.
