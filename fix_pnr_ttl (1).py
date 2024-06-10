# Databricks notebook source
# MAGIC %md 
# MAGIC <img width="800" height="0"  src="https://www.etihad.com/content/dam/eag/etihadairways/etihadcom/Global/logo/header/header-text-image-web.svg" alt="Etihad Airways Logo"  >

# COMMAND ----------

# MAGIC %md
# MAGIC # Objective
# MAGIC - **To Anlysis and test a suitable model to predict the agency behavior like cancel,ticketed etc. based on the history data on certain set of PNR**
# MAGIC

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

dbutils.fs.ls("/mnt/ainavigatorpath/")

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/mnt/ainavigatorpath/ai_hackathon_data-v2.csv")


# COMMAND ----------

from pyspark.sql import functions as F
df_n_full = df.withColumn("pnr_creation_datetime1",F.to_timestamp("pnr_creation_dateTime", "yyyy-MM-dd'T'HH:mm:ss'Z'") ).withColumn("ticket_issue_date1",F.to_timestamp("ticket_issue_date", "yyyy-MM-dd") )
df_n_full_pos = df_n_full.withColumn("date_diff", F.date_diff("ticket_issue_date1","pnr_creation_datetime1"))
df_p = df_n_full_pos.filter(F.col("date_diff")>=0).drop("date_diff","ticket_issue_date1","pnr_creation_datetime1")
df_n = df_n_full_pos.filter(F.col("date_diff")<0)

# COMMAND ----------

df_p.createOrReplaceTempView("ai_hackathon_data_v1")
spark.sql("CREATE or Replace TABLE ai_hackathon_data_v1 USING delta AS SELECT * FROM ai_hackathon_data_v1")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ai_hackathon_data_v1

# COMMAND ----------

# MAGIC %md
# MAGIC #Label Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC #Prep Training Data

# COMMAND ----------

df_base = spark.sql("select * from ai_hackathon_data_v1")


# COMMAND ----------

from datetime import date
date_var = date(2023, 12, 31)
print(date_var)
df_base_before = df_base.filter(F.col("pnr_creation_dateTime")<= date_var )


# COMMAND ----------

df_base_pd = pd.DataFrame(df_base_before.toPandas())



# COMMAND ----------


# df = pd.DataFrame(data) 

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(df_base_pd)

# Print original DataFrameprint("Original DataFrame:")
print(df) 
# df.dtypes

encoding_mapping = {}
# Initialize LabelEncoder
label_encoder = LabelEncoder() 
#  Fit and transform the  data  pd.DataFrame(df_standard_scaled, columns=df.columns)
for i in df.columns:
    df[i] = label_encoder.fit_transform(df[i]) 
    encoding_mapping[i] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
 # Print the DataFrame with encoded values
print("DataFrame with Encoded Education Levels:")
print(df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Scaling

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
 
# # Sample DataFrame
# data = {'Feature1': [10, 20, 30, 40, 50],
#         'Feature2': [100, 200, 300, 400, 500],
#         'Feature3': [1000, 2000, 3000, 4000, 5000]}
df = pd.DataFrame(df)
 
# print("Original DataFrame:")
# print(df)
 
# RobustScaler
robust_scaler = RobustScaler()
df_robust_scaled = robust_scaler.fit_transform(df)   #df_base_train_data_pd
df_robust_scaled = pd.DataFrame(df_robust_scaled, columns=df.columns)
# print("Robust Scaled DataFrame:")
# print(df_robust_scaled)

# COMMAND ----------

df_robust_scaled_train_x =  df_robust_scaled.drop("agency_behavour", axis= 1)
df_robust_scaled_train_y =  df_robust_scaled["agency_behavour"]

# COMMAND ----------

# MAGIC %md
# MAGIC #Prep Test Data

# COMMAND ----------

df_base_after = df_base.filter(F.col("pnr_creation_dateTime")> date_var )

# COMMAND ----------


df_base_pd = df_base_after.toPandas()


# df = pd.DataFrame(data) 

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(df_base_pd) 
# Print original DataFrameprint("Original DataFrame:")
# print(df) 
# Initialize LabelEncoder
label_encoder = LabelEncoder() 
#  Fit and transform the  data  pd.DataFrame(df_standard_scaled, columns=df.columns)
for i in df.columns:
    df[i] = label_encoder.fit_transform(df[i]) 
 # Print the DataFrame with encoded values

# # df["agency_behavour_l"] = df["agency_behavour"]
# print("DataFrame with Encoded Education Levels:")
# print(df)


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
 
# # Sample DataFrame
# data = {'Feature1': [10, 20, 30, 40, 50],
#         'Feature2': [100, 200, 300, 400, 500],
#         'Feature3': [1000, 2000, 3000, 4000, 5000]}
df_base_test_data_pd = df
df = pd.DataFrame(df_base_test_data_pd)
 
# print("Original DataFrame:")
# print(df)

robust_scaler = RobustScaler()
df_robust_scaled_test = robust_scaler.fit_transform(df)
df_robust_scaled_test = pd.DataFrame(df_robust_scaled_test, columns=df.columns)
# print("Robust Scaled DataFrame:")
# print(df_robust_scaled_test)


df_robust_scaled_test_x =  df_robust_scaled_test.drop("agency_behavour", axis= 1)
df_robust_scaled_test_y =  df_robust_scaled_test["agency_behavour"]


# COMMAND ----------

# MAGIC %md
# MAGIC #Model Exploration

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=100),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
    # 'XGBoost': XGBClassifier()
    # 'Support Vector Machine': SVC(max_iter=100),
    # 'k-Nearest Neighbors': KNeighborsClassifier()
}



 
X_train = df_robust_scaled_train_x
y_train = df_robust_scaled_train_y
X_test = df_robust_scaled_test_x
y_test = df_robust_scaled_test_y


# Train and evaluate models
results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'model': model,
            'y_pred' :  y_pred

        }
        print(f"passed for {name}")
    except:
        print(f"failing in {name}")
 
# Display results
results_df = pd.DataFrame(results)
print(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Metric

# COMMAND ----------

results_df

# COMMAND ----------

df_adding = [row for row in df_base_after.collect()]

list_of_lists = [[value for value in row] for row in df_base_after.collect()]

# COMMAND ----------

from pyspark.sql.types import StructType,StructField,StringType,DateType,TimestampType,IntegerType,DecimalType,DoubleType,LongType,FloatType

up_schema = StructType([StructField('mask_pnr', StringType(), True), StructField('pnr_creation_dateTime', StringType(), True), StructField('pnr_keywords_code', StringType(), True), StructField('pnr_keywords_text', StringType(), True), StructField('ticket_issue_date', StringType(), True), StructField('TTL_Date', StringType(), True), StructField('pnr_AutomatedProcesses_code', StringType(), True), StructField('cancellation_Date', StringType(), True), StructField('Number_of_PAX', StringType(), True), StructField('Booking_System_Code', StringType(), True), StructField('agent_type', StringType(), True), StructField('pnr_creation_pointOfSale_office_id', StringType(), True), StructField('segid', StringType(), True), StructField('Marketing_Airline_Code', StringType(), True), StructField('Marketing_Flight_Number', StringType(), True), StructField('Op_Airline_Code', StringType(), True), StructField('Op_flightNumber', StringType(), True), StructField('RBD', StringType(), True), StructField('no_days_ticked_TTL_bf_exp', StringType(), True), StructField('no_days_cancel_TTL_bf_exp', StringType(), True), StructField('no_days_cancel_TTL_af_bking', StringType(), True), StructField('no_days_cancel_bf_std', StringType(), True), StructField('Cancelling_close_TTL_exp', StringType(), True), StructField('agency_behavour', StringType(), True), StructField('agency_behavour_y_predict', StringType(), True)])

# COMMAND ----------

y_p_value =  results['Random Forest']['y_pred']
for i in range(len(y_p_value)):
    list_of_lists[i].append(str(y_p_value[i]))



# COMMAND ----------

df_final = spark.createDataFrame(data = list_of_lists,schema = up_schema)
df_final.display()

# COMMAND ----------

df_final_full = df_final.withColumn("derived_col_predict", F.when(F.col("agency_behavour_y_predict") == '0.0',"ticketed").when(F.col("agency_behavour_y_predict") == '-4.0',"auto cancel").when(F.col("agency_behavour_y_predict") == '-1.0',"cancel earlier than 7days").when(F.col("agency_behavour_y_predict") == '-2.0',"cancel 1-3 day of exp").when(F.col("agency_behavour_y_predict") == '1.0',"ticketed and cancelled").when(F.col("agency_behavour_y_predict") == '-3.0',"average").when(F.col("agency_behavour_y_predict") == '2.0',"None").otherwise("unmapped"))

df_final_full.display()

# COMMAND ----------

df_final_full_actual = df_final_full.select("agency_behavour").groupBy("agency_behavour").agg(F.count(F.col("agency_behavour")).alias("agency_bh_actual_count"))
df_final_full_actual.display()

# COMMAND ----------

df_final_full_predict = df_final_full.select("derived_col_predict").withColumnRenamed("derived_col_predict", "agency_behavour").groupBy("agency_behavour").agg(F.count(F.col("agency_behavour")).alias("derived_col_predict_actual_count"))
df_final_full_predict.display()

# COMMAND ----------

df_resultant =  df_final_full_actual.join(df_final_full_predict,["agency_behavour"],"left").withColumnRenamed("agency_behavour", "agency_behaviour")
df_resultant.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Prediction Outcomes

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert Spark DataFrame to Pandas
pandas_df = df_resultant.toPandas()

# Set the figure size
plt.figure(figsize=(15, 6))

# Create a list of unique agency_behaviours
agency_behaviours = pandas_df['agency_behaviour'].unique()

# For each agency_behaviour, create a pair of bars
for i, agency_behaviour in enumerate(agency_behaviours):
    # Filter data for the current agency_behaviour
    data = pandas_df[pandas_df['agency_behaviour'] == agency_behaviour]
    
    # Create a pair of bars
    plt.bar(i - 0.2, data['agency_bh_actual_count'], 0.2, color='b')
    plt.bar(i + 0.2, data['derived_col_predict_actual_count'], 0.2, color='r')

# Set the x-ticks to be the agency_behaviours
plt.xticks(range(len(agency_behaviours)), agency_behaviours)

# Add labels and title
plt.xlabel('Agency Behaviour')
plt.ylabel('Count')
plt.title('Comparison of Actual Count and Predicted Count for Combine Agency Behaviour')
plt.legend(['Agency BH Actual Count', 'Derived Col Predict Actual Count'])

# Display the plot
plt.show()


# COMMAND ----------

# MAGIC %md 
# MAGIC # Conclusion
# MAGIC - Some Variable like tickted, cancel earlier than 7 days , average are more closure to accuracy than rest

# COMMAND ----------

# MAGIC %md 
# MAGIC # Way Forward
# MAGIC - Feature engineering agency wise can be done that include building more dependent varaible that decide performace
# MAGIC - Other Model to be tested
# MAGIC - Source Date can be more refined and less niosy. 
