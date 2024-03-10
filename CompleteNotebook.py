#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[38]:


from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import lead, avg, mean, stddev, countDistinct, when, col, udf, month, lit, median as _median
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, BinaryClassificationMetrics
from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier


import xgboost as xgb
import shap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# # PySpark Initialization

# In[39]:




spark = SparkSession.builder        .master("local")        .appName("GSOD")        .config('spark.ui.port', '4050')        .getOrCreate()

spark


# # Data Loading

# In[40]:


df = spark.read.format("csv").load("gsod-2023.csv", header=True, inferSchema=True)
df.printSchema()


# In[41]:


df.show(5)


# In[42]:



# List of columns with their respective missing value indicators
missing_value_indicators = {
    'temp': 9999.9, 'dewp': 9999.9, 'slp': 9999.9, 'stp': 9999.9,
    'visib': 999.9, 'wdsp': 999.9, 'mxpsd': 999.9, 'gust': 999.9,
    'max': 9999.9, 'min': 9999.9, 'prcp': 99.99, 'sndp': 999.9,
}

# Replace placeholder values with None (which represents null in PySpark)
for column, indicator in missing_value_indicators.items():
    df = df.withColumn(column, when(col(column) == indicator, None).otherwise(col(column)))


# In[43]:


# Assuming other columns have missing values represented by 'null' or 'None'
missing_counts = {column: df.filter(col(column).isNull()).count() for column in df.columns}

# View the count of missing values for all columns in df
for column, count in missing_counts.items():
    print(f"{column}: {count} missing values")


# # Visualization

# In[44]:


# For the plot, use Seaborn to plot the distribution of the first numerical feature
sns.set_style("whitegrid")
feature_to_plot = 'temp'
pdf = df.select(feature_to_plot).toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(pdf[feature_to_plot].dropna(), kde=False)
plt.title(f'Distribution of {feature_to_plot}')
plt.xlabel(feature_to_plot)
plt.ylabel('Frequency')
plt.show()


# # Feature Selection

# In[45]:


# List of columns to drop
columns_to_drop = ['dewp', 'wban', 'gust', 'flag_max', 'flag_min', 'flag_prcp', 'sndp', 'slp']

# Drop the columns
df = df.drop(*columns_to_drop)


# In[46]:


numerical_columns = [
    'temp',           # Mean temperature
    # 'dewp',           # Mean dew point
    # 'slp',            # Mean sea level pressure
    'stp',            # Mean station pressure
    'visib',          # Mean visibility
    'wdsp',           # Mean wind speed
    'mxpsd',          # Maximum sustained wind speed
    'max',            # Maximum temperature
    'min',            # Minimum temperature
    'prcp',           # Total precipitation
    # Number of values that were used to compute the mean
    'count_temp',     
    'count_dewp',     
    'count_slp',      
    'count_stp',      
    'count_visib',    
    'count_wdsp'      
]

for column in numerical_columns:
    # Calculate mean, standard deviation, and distinct count
    stats = df.select(
        mean(col(column)).alias('mean'),
        stddev(col(column)).alias('stddev'),
        countDistinct(col(column)).alias('distinct_count')
    ).collect()[0]
    
    # Calculate median
    median_value = df.stat.approxQuantile(column, [0.5], 0.05)[0]

    # Calculate mode - mode may not be accurate if the dataset is very large due to Spark's distributed nature.
    mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[column]

    # Print the statistics in one line
    print(f"{column} | Mean: {stats['mean']:.2f} | StdDev: {stats['stddev']:.2f} | Median: {median_value:.2f} | Mode: {mode_value} | Distinct Count: {stats['distinct_count']}")


# In[47]:



categorical_columns = [
    'fog',                  # Fog 
    'rain_drizzle',         # Rain or drizzle 
    'snow_ice_pellets',     # Snow or ice pellets 
    'hail',                 # Hail 
    'thunder',              # Thunder
    'tornado_funnel_cloud', # Tornado or funnel cloud 
    # Flags are categorical and indicate data quality or source
    # 'flag_max',             
    # 'flag_min',             
    # 'flag_prcp',            
    # Time-related fields can be used for time series analysis but are not direct features themselves
    # 'year',                 
    # 'mo',                   
    # 'da'                    
]

# For categorical features, count the number of distinct categories
for column in categorical_columns:
    distinct_count = df.select(countDistinct(col(column)).alias('distinct_count')).collect()
    print(f"{column} | Unique Categories: {distinct_count[0]['distinct_count']}")


# # Data Imputation

# In[48]:


def MedianImputer(df, column_name):
    """
    Imputes missing values in a DataFrame column with the median value grouped by the station,
    and uses the global median as a fallback if all values for a feature at a station are None.
    Reports the number of values imputed using station-specific and global median.

    :param df: Spark DataFrame containing the column to impute.
    :param column_name: Name of the column to impute.
    """
    
    initial_missing_count = df.filter(col(column_name).isNull()).count()

    # Calculate global median for the fallback
    global_median = df.agg(_median(col(column_name)).alias('global_median')).collect()[0]['global_median']

    # Group by 'stn' and calculate median for each group
    median_values = df.groupBy('stn').agg(_median(col(column_name)).alias('median')).collect()

    # Initialize counter for imputation tracking
    station_imputed_count = 0
    global_imputed_count = 0

    # Create a dictionary to map each 'stn' to its median value, including a global fallback
    median_map = {}
    for row in median_values:
        if row['median'] is not None:
            median_map[row['stn']] = row['median']
        else:
            median_map[row['stn']] = global_median
            global_imputed_count += df.filter((col('stn') == row['stn']) & col(column_name).isNull()).count()

    # Use global median if there are stations not covered in median_values
    median_map['global'] = global_median

    # User defined function to return the median based on the 'stn' value, falling back to global median
    def get_median(stn):
        return median_map.get(stn, global_median)

    # Register the function as a UDF
    median_udf = udf(get_median, FloatType())

    # Apply the UDF to the DataFrame to impute missing values, tracking the number of imputations
    df_with_imputation_flag = df.withColumn(column_name+'_imputed', when(col(column_name).isNull(), median_udf(col('stn'))).otherwise(col(column_name)))                                 .withColumn("imputed_flag", when(col(column_name).isNull(), 1).otherwise(0))

    station_imputed_count = initial_missing_count - global_imputed_count

    # Drop the original column and rename the imputed column to the original column name, remove the imputation flag
    df = df_with_imputation_flag.drop(column_name)                                 .drop("imputed_flag")                                 .withColumnRenamed(column_name+'_imputed', column_name)

    print(f"Imputed {station_imputed_count} missing values using station median and {global_imputed_count} missing values using global median in '{column_name}'.")

    return df




def ProximityMedian(df, column, initial_num_days, max_days, fallback_strategy='median'):
    """
    Imputes missing values in a DataFrame column by taking the average of the values
    num_days before and after the date of the missing observation, for the same station.
    The num_days is increased progressively until there are no more missing values 
    or until the max_days is reached. If imputation still fails after reaching max_days, 
    a fallback strategy is applied.

    :param df: Spark DataFrame containing the column to impute.
    :param column: Name of the column to impute.
    :param initial_num_days: Initial number of days before and after to consider for the average.
    :param max_days: Maximum number of days to increase the window for imputation.
    :param fallback_strategy: Strategy to use if the imputation is not successful ('median' or 'mean').
    :returns: DataFrame with the imputed column.
    """
    num_days = initial_num_days
    original_missing_count = df.filter(col(column).isNull()).count()
    
    # Perform the proximity mean imputation
    while num_days <= max_days and original_missing_count > 0:
        windowSpec = Window.partitionBy('stn').orderBy('date').rowsBetween(-num_days, num_days)
        df = df.withColumn(
            f"{column}_imputed",
            when(
                col(column).isNull(),
                avg(col(column)).over(windowSpec)
            ).otherwise(col(column))
        )
        
        missing_count_after = df.filter(col(f"{column}_imputed").isNull()).count()
        if missing_count_after == 0:
            break
        num_days *= 2  # Increase the window size for the next iteration

    # If there are still missing values after the maximum window size is reached, use the fallback strategy
    if missing_count_after > 0:
        fallback_value = df.stat.approxQuantile(column, [0.5], 0.001)[0] if fallback_strategy == 'median' else df.agg(avg(col(column))).first()[0]
        df = df.withColumn(
            f"{column}_imputed",
            when(
                col(f"{column}_imputed").isNull(),
                lit(fallback_value)
            ).otherwise(col(f"{column}_imputed"))
        )
    
    final_df = df.drop(column).withColumnRenamed(f"{column}_imputed", column)
    
    # Print a summary of what happened
    print(f"ProximityMedian imputation for '{column}':"
          f" Initial Missing: {original_missing_count},"
          f" Remaining Missing (after ProximityMedian): {missing_count_after},"
          f" Filled the rest with strategy: '{fallback_strategy}'.")

    return final_df


def ImputeTempWithSeasonalMedian(df, column, initial_num_days=7, max_days=31):
    """
    Imputes missing values for a given temperature column ('max' or 'min') in a DataFrame
    by using the median temperature value for the corresponding station ('stn') and month.
    If this fails to impute all missing values, falls back to the ProximityMean method.

    Args:
    df (DataFrame): The Spark DataFrame containing weather data.
    column (str): The name of the temperature column to impute ('max' or 'min').
    initial_num_days (int): Starting number of days for proximity mean imputation.
    max_days (int): Maximum number of days for expanding the window in proximity mean imputation.

    Returns:
    DataFrame: The DataFrame with missing temperature values imputed.
    """

    initial_missing_count = df.filter(col(column).isNull()).count()
    
    # Calculate median temperature grouped by 'stn' and 'mo' (month)
    median_temp = df.groupBy('stn', month(df.date).alias('mo')).agg(_median(col(column)).alias('median_temp'))
    
    # Join the median_temp DataFrame with the original DataFrame based on 'stn' and 'mo'
    df = df.join(median_temp, (df.stn == median_temp.stn) & (month(df.date) == median_temp.mo), 'left_outer')

    # Impute missing values with the corresponding median_temp
    df = df.withColumn(column, when(col(column).isNull(), col('median_temp')).otherwise(col(column)))

    final_missing_count = df.filter(col(column).isNull()).count()
    
    print(f"Imputed {initial_missing_count - final_missing_count} missing values in '{column}' using seasonal median by station and month.")

    # Drop the extra columns used for the join
    df = df.drop(median_temp.stn).drop(median_temp.mo).drop('median_temp')

    if final_missing_count != 0:
        # Fallback to ProximityMean if not all values were imputed
        df = ProximityMedian(df, column, initial_num_days, max_days)
        print(f"Used ProximityMedian as a fallback for remaining {final_missing_count} missing values.")
    
    return df


# In[49]:


imputation_strategy = {
    # 'dewp': lambda df: MedianImputer(df, 'dewp'),
    # 'slp': lambda df: MedianImputer(df, 'slp'),
    'visib': lambda df: ProximityMedian(df, 'visib', initial_num_days=7, max_days=30, fallback_strategy='median'),
    'wdsp': lambda df: ProximityMedian(df, 'wdsp', initial_num_days=7, max_days=30, fallback_strategy='median'),
    'mxpsd': lambda df: ProximityMedian(df, 'mxpsd', initial_num_days=7, max_days=30, fallback_strategy='median'),
    'prcp': lambda df: df.na.fill({'prcp': 0}),  # Zero imputation for precipitation
    'max': lambda df: ImputeTempWithSeasonalMedian(df, 'max'),
    'min': lambda df: ImputeTempWithSeasonalMedian(df, 'min')
}


# In[50]:


for column, impute_function in imputation_strategy.items():
    df = impute_function(df)


# In[51]:


df.show(5)


# In[52]:


# Assuming other columns have missing values represented by 'null' or 'None'
missing_counts_2 = {column: df.filter(col(column).isNull()).count() for column in df.columns}

# View the count of missing values for all columns in df
for column, count in missing_counts_2.items():
    print(f"{column}: {count} missing values")


# In[53]:


df.printSchema()


# # Vectorization

# In[54]:


# Process categorical columns with StringIndexer and OneHotEncoder
for categorical_col in categorical_columns:
    string_indexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index")
    df = string_indexer.fit(df).transform(df)
    encoder = OneHotEncoder(inputCol=categorical_col + "_index", outputCol=categorical_col + "_vec")
    df = encoder.fit(df).transform(df)


# In[55]:


# Create a window spec
windowSpec = Window.partitionBy("stn").orderBy("date")

# For regression, shift 'max' temperature by one day to predict next day's temperature
df = df.withColumn("next_day_max", lead("max").over(windowSpec))

# For classification, shift 'rain_drizzle' by one day to predict rain/drizzle for the next day
df = df.withColumn("next_day_rain", lead("rain_drizzle").over(windowSpec))

# Filter out the rows with null 'next_day_max' or 'next_day_rain' as these will be our labels
df_filtered = df.filter(df.next_day_max.isNotNull() & df.next_day_rain.isNotNull())

# Assemble numerical features
numerical_assembler = VectorAssembler(inputCols=numerical_columns, outputCol="numerical_features")
df_numerical_assembled = numerical_assembler.transform(df_filtered)


# In[56]:


df_numerical_assembled.show(3)


# In[57]:


# StandardScaler for numerical features
scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features")
df_scaled_numerical = scaler.fit(df_numerical_assembled).transform(df_numerical_assembled)

# Assemble scaled numerical features with one-hot encoded categorical features
final_assembler_inputs = [c + "_vec" for c in categorical_columns] + ["scaled_numerical_features"]
final_assembler = VectorAssembler(inputCols=final_assembler_inputs, outputCol="features")
df_final = final_assembler.transform(df_scaled_numerical)


# In[58]:


df_final.show(3)


# # Train-Test Split

# In[59]:


# Split the data into training and test sets
(train_data, test_data) = df_final.randomSplit([0.8, 0.2])


# In[60]:


train_data.select('features').take(3)


# # ML - Regression - Predict next day max temperature 

# In[61]:


regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="next_day_max", metricName="rmse")


# ## Linear Regression (baseline)

# In[62]:


# Regression model: predict next day's maximum temperature
regressor = LinearRegression(featuresCol="features", labelCol="next_day_max")
regression_model = regressor.fit(train_data)

# Linear Regression - Predict and Evaluate
lr_predictions = regression_model.transform(test_data)
lr_rmse = regression_evaluator.evaluate(lr_predictions)
print(f"Linear Regression RMSE: {lr_rmse}")


# In[63]:


lr_predictions.show(2)


# In[64]:


predictions_and_labels_rdd = lr_predictions.select(col("prediction"), col("next_day_max")).rdd.map(lambda row: (row[0], row[1]))

# Instantiate RegressionMetrics based on the RDD
regression_metrics = RegressionMetrics(predictions_and_labels_rdd)

# Now you can use various metrics provided by RegressionMetrics
print(f"RMSE: {regression_metrics.rootMeanSquaredError}")
print(f"MSE: {regression_metrics.meanSquaredError}")
print(f"MAE: {regression_metrics.meanAbsoluteError}")
print(f"R2: {regression_metrics.r2}")


# ## XGBoostRegressor

# In[65]:


# XGBoost Regression - Train and Evaluate
xgboost_regressor = SparkXGBRegressor(features_col="features", label_col="next_day_max")
xgb_regression_model = xgboost_regressor.fit(train_data)
xgbr_predictions = xgb_regression_model.transform(test_data)
# xgb_lr_rmse = regression_evaluator.evaluate(xgbr_predictions)
# print(f"XGBoost Regression RMSE: {xgb_lr_rmse}")


# In[67]:


predictions_and_labels_rdd = xgbr_predictions.select(col("prediction"), col("next_day_max")).rdd.map(lambda row: (row[0], row[1]))

# Instantiate RegressionMetrics based on the RDD
regression_metrics = RegressionMetrics(predictions_and_labels_rdd)

# Now you can use various metrics provided by RegressionMetrics
print(f"RMSE: {regression_metrics.rootMeanSquaredError}")
print(f"MSE: {regression_metrics.meanSquaredError}")
print(f"MAE: {regression_metrics.meanAbsoluteError}")
print(f"R2: {regression_metrics.r2}")


# # ML - Classification - Predict next day rain (binary)

# In[68]:


# For Classification Evaluation (Random Forest and XGBoost)
classification_evaluator = MulticlassClassificationEvaluator(labelCol="next_day_rain", predictionCol="prediction", metricName="accuracy")


# ## Random Forest Classifier (baseline)

# In[ ]:


# # Classification model: predict if there will be rain/drizzle the next day
# classifier = RandomForestClassifier(featuresCol="features", labelCol="next_day_rain")
# classification_model = classifier.fit(train_data)
# # Random Forest Classifier - Predict and Evaluate
# rf_predictions = classification_model.transform(test_data)
# rf_accuracy = classification_evaluator.evaluate(rf_predictions)
# print(f"Random Forest Classifier Accuracy: {rf_accuracy}")


# In[ ]:


# # Random Forest feature importances
# rf_feature_importances = classification_model.featureImportances.toArray()
# print("Random Forest Feature Importances:")
# for i, importance in enumerate(rf_feature_importances):
#     print(f"Feature {i + 1}: {importance}")


# ## XGBoostClassifier

# In[69]:


# XGBoost Classification - Train and Evaluate
xgboost_classifier = SparkXGBClassifier(features_col="features", label_col="next_day_rain")
xgb_classification_model = xgboost_classifier.fit(train_data)
xgbc_predictions = xgb_classification_model.transform(test_data)


# In[74]:


xgbc_accuracy = classification_evaluator.evaluate(xgbc_predictions)
predictions_and_labels_rdd = xgbc_predictions.select(
    col("prediction").cast("double"),  # Cast prediction to DoubleType
    col("next_day_rain").cast("double")  # Cast next_day_rain to DoubleType
).rdd.map(lambda row: (row[0], row[1]))

# Instantiate RegressionMetrics based on the RDD
binary_classification_metrics = BinaryClassificationMetrics(predictions_and_labels_rdd)

# Now you can use various metrics provided by RegressionMetrics
print(f"XGBoost Classifier Accuracy: {xgbc_accuracy}")
print(f"areaUnderPR: {binary_classification_metrics.areaUnderPR}")
print(f"areaUnderROC: {binary_classification_metrics.areaUnderROC}")


# ## SHAP for Classification

# In[75]:


# Save the PySpark XGBoost model
model_path = "./xgb_classification_model"
xgb_classification_model.write().overwrite().save(model_path)

# Load the model using Python's XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model(model_path + '/model/part-00000')

# Convert the test data to Pandas DataFrame
X_test_pandas = test_data.select("features").toPandas()
print(X_test_pandas.head())


# In[ ]:


# Assume 'numerical_columns' is a list of your original numerical feature names
scaled_feature_names = ["scaled_" + col for col in numerical_columns]

# Categorical feature names after one-hot encoding
categorical_feature_vec_names = [c + "_vec" for c in categorical_columns]

# Final feature names list including both categorical and numerical features
final_feature_names = categorical_feature_vec_names + scaled_feature_names

# Now create the Pandas DataFrame with the final feature names
feature_array = np.array(X_test_pandas['features'].tolist())
feature_df = pd.DataFrame(feature_array, columns=final_feature_names)


# In[78]:


# Extract feature names from the DataFrame schema
feature_names = [col['name'] for col in df_final.schema["features"].metadata["ml_attr"]["attrs"]["binary"]] +                 [col['name'] for col in df_final.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]]

# Convert the feature vector column in Spark DataFrame to a Pandas DataFrame with proper feature names
feature_array = np.array(X_test_pandas['features'].tolist())
feature_df = pd.DataFrame(feature_array, columns=feature_names)

# Proceed with the DMatrix creation and SHAP values computation
X_test_matrix = xgb.DMatrix(feature_df)


# In[79]:


# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_matrix)


# In[84]:


# Summarize the effects of all the features
shap.summary_plot(shap_values, feature_df)


# In[ ]:




