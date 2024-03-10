# Weather Data Analysis with PySpark

## Overview

This project analyzes a comprehensive weather dataset using Apache PySpark. It demonstrates data loading, preprocessing, feature engineering, and machine learning to predict the next day's maximum temperature and the likelihood of rain. This project efficiently handles large datasets, performs exploratory data analysis (EDA), data imputation, visualization, and predictive modeling using regression and classification techniques.

## Technologies Used

- PySpark for distributed data processing
- XGBoost for regression and classification models
- SHAP for model interpretability
- Matplotlib and Seaborn for data visualization
- Pandas and NumPy for data manipulation

## Features

- Data cleaning and preprocessing to handle missing values and outliers
- Feature engineering to prepare the dataset for machine learning models
- Regression analysis to predict the maximum temperature of the next day
- Binary classification to predict the occurrence of rain the next day
- Evaluation of machine learning models with metrics such as RMSE and accuracy
- Visualization of data distributions and model predictions
- Implementation of various data imputation strategies for handling missing values

## Dataset

The project uses the Global Surface Summary of the Day (GSOD) dataset from Google BigQuery, which contains approximately 4 million rows with numerous missing values, necessitating extensive data imputation. 

## Data Imputation Strategies

Implemented data imputation strategies include:
- Median Imputation: Imputes missing values using the median of the column.
- Proximity Median: Imputes missing values based on a median within a window of dates for the same station.
- Zero Imputation: Specifically for precipitation, missing values are assumed to be days with no precipitation and are set to zero.
- Seasonal Median Imputation: For temperature columns, missing values are imputed using the median temperature for the station and month, providing a seasonally adjusted imputation.

## Requirements

Ensure you have Java 8 or 11 installed and Apache Spark set up on your system.

### Setup and Installation

1. Install the Python dependencies from the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

2. **Download the GSOD Dataset**: To access the GSOD dataset, download it from Google BigQuery. The dataset contains approximately 4 million rows with numerous missing values, requiring comprehensive data imputation.

## How to Use

1. **Initialize PySpark Session**: Start by initializing a Spark session.
2. **Load the Dataset**: Load the GSOD dataset into a Spark DataFrame. Adjust the path to your dataset file accordingly.
3. **Data Preprocessing**: Perform data cleaning and preprocessing steps to handle missing values and outliers.
4. **Feature Engineering**: Execute feature selection and engineering to prepare the data for modeling.
5. **Model Training and Evaluation**: Train regression and classification models. Evaluate the models using appropriate metrics.
6. **Visualization**: Use Matplotlib and Seaborn for data distributions and model predictions.
7. **SHAP Analysis**: Compute SHAP values for model interpretation.


## License

This project is open source and available under the [MIT License](LICENSE.md).
