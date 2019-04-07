# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### In this notebook, you can learn how to use Spark SQL for big data analysis on SF crime data. (https://data.sfgov.org/Public-Safety/sf-data/skgt-fej3/data). 
# MAGIC The first part is OLAP for scrime data analysis.  
# MAGIC The second part is unsupervised learning for spatial data analysis.   
# MAGIC 
# MAGIC **Note**: you can download the small data (one month e.g. 2018-10) for debug, then download the data from 2013 to 2018 for testing and analysising. 

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
import warnings
plt.style.use('ggplot')

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

# MAGIC %md Since the Data of 2015~2017 and data of 2018 are seperate files in different format, so I read them seperately into DataFrame `data_17` and `data_18`

# COMMAND ----------

# DBTITLE 0,option 1 to get dataframe and sql
from pyspark.sql.functions import col
from pyspark.sql.functions import hour, date_format, to_date
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

data_17 = spark.read.format("csv").option("header", "true").load("/FileStore/tables/sfdata_15_17.csv")
data_17 = data_17.withColumn('X', data_17["X"].cast('double')) \
                 .withColumn('Y', data_17["Y"].cast('double')) \
                 .withColumn("day",to_date(col("Date"), "MM/dd/yy")) \
                 .withColumn('hour', hour(data_17['Time'])) \


data_17 = data_17.drop('IncidntNum','Descript','Address','PdId','Time','Date') \
                 .withColumnRenamed('day', 'Date')

data_18 = spark.read.format("csv").option("header", "true").load("/FileStore/tables/sfdata_18.csv")

data_18 = data_18.drop('Report Datetime','Row ID','Incident ID','Incident Number','CAD Number','Report Type Code',\
                       'Report Type Description','Filed Online','Incident Code','Incident Subcategory','Incident Description',\
                       'Intersection','CNN','Analysis Neighborhood','Supervisor District')

data_18 = data_18.withColumn('X', data_18['Longitude'].cast('double')) \
                 .withColumn('Y', data_18['Latitude'].cast('double')) \
                 .withColumn("Date",to_date(col("Incident Date"), "yyyy/MM/dd")) \
                 .withColumn('hour', hour(data_18['Incident Time'])) \
                 .drop('Longitude','Latitude','Incident Date','Incident Time','Incident Datetime')

data_18 = data_18.withColumnRenamed('Incident Day of Week', 'DayOfWeek') \
                 .withColumnRenamed('Incident Year','year') \
                 .withColumnRenamed('Incident Category','Category') \
                 .withColumnRenamed('Police District','District')

# COMMAND ----------

data_18.count() + data_17.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### OLAP: 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.

# COMMAND ----------

q1_17 = data_17.groupBy('category').count().orderBy('count', ascending=False)
q1_17 = q1_17.toPandas()

q1_18 = data_18.groupBy(data_18['category'].alias('category')).count().orderBy('count', ascending=False)
q1_18 = q1_18.toPandas()

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
axes[0].bar(range(1,20), q1_17['count'][0:19], align='center')
axes[1].bar(range(1,20), q1_18['count'][0:19], align='center')
axes[0].set_xticks(range(1,20))
axes[0].set_xticklabels(q1_17['category'][0:19],rotation=45,fontsize=6)
axes[1].set_xticks(range(1,20))
axes[1].set_xticklabels(q1_18['category'][0:19],rotation=90,fontsize=6)
# fig.subplots_adjust(bottom=0.2)
axes[0].set_title("Number of Crime by Category(top 20) from 2015-2017")
axes[1].set_title("Number of Crime by Category(top 20) in 2018")

fig.tight_layout()
display(fig)

# COMMAND ----------

# DBTITLE 0,Q2 question (OLAP) Counts the number of crimes for different district, and visualize your results
# MAGIC %md
# MAGIC #### OLAP
# MAGIC Counts the number of crimes for different district, and visualize your results

# COMMAND ----------

from pyspark.sql.functions import year

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
q2 = []
years_plot = range(2015,2019)
for i in range(0,4):
  if i < 3:
    q2.append(data_17.filter(year(col('Date'))==years_plot[i]) \
              .groupBy(data_17['PdDistrict'].alias('district')) \
              .count().orderBy('count',ascending=False).dropna() \
              .toPandas()
             )
  if i == 3:
    q2.append(data_18.groupBy(data_18['District'].alias('district')) \
              .count().orderBy('count',ascending=False).dropna().toPandas()
             )
    
  sns.barplot(x='district',y='count',data=q2[i], ax = axes[i//2][i%2])
  axes[i//2][i%2].set_xticklabels(q2[i]['district'],rotation=45,fontsize=7)
  #axes[i//2][i%2].tick_params(labelbottom=True)
  axes[i//2][i%2].set_ylabel('Number of Crimes')
  axes[i//2][i%2].set_title('Number of Crimes in {0}'.format(years_plot[i]))
display()

# COMMAND ----------

# Old method for q2 (Deprecated)
# q2_17 = data_17.groupBy(data_17['PdDistrict'].alias('district')).count().orderBy('count',ascending=False).dropna().toPandas()
# q2_18 = data_18.groupBy(data_18['District'].alias('district')).count().orderBy('count',ascending=False).dropna().toPandas()

# fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
# axes[0].bar(range(1,q2_17.shape[0]+1), q2_17['count'], align='center')
# axes[1].bar(range(1,q2_18.shape[0]+1), q2_18['count'], align='center')

# axes[0].set_xticks(range(1,q2_17.shape[0]+1))
# axes[0].set_xticklabels(q2_17['district'],rotation=45,fontsize=7)

# axes[1].set_xticks(range(1,q2_18.shape[0]+1))
# axes[1].set_xticklabels(q2_18['district'],rotation=45,fontsize=8)
# # fig.subplots_adjust(bottom=0.2)
# axes[0].set_title("Number of Crime by District from 2015-2017")
# axes[1].set_title("Number of Crime by District in 2018")

# fig.tight_layout()
# display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### OLAP
# MAGIC Count the number of crimes each "Sunday" at "SF downtown". 
# MAGIC hints: SF downtown is defiend  via the range of spatial location. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/

# COMMAND ----------

from pyspark.sql.functions import year
import matplotlib.dates as mdates
monthsFmt = mdates.DateFormatter('%m')

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,10))
df = []
years = range(2015,2019)
for i in range(0,4):
  if i < 3:
    df.append(data_17.filter(data_17.X > -122.41) \
                     .filter(data_17.X < -122.40) \
                     .filter(data_17.Y > 37.79) \
                     .filter(data_17.Y < 37.80) \
                     .filter(year(col('Date'))==2015) \
                     .filter(col('DayOfWeek')=='Sunday') \
                     .groupBy('Date').count().orderBy('Date') \
                     .toPandas())
  else:
    df.append(data_18.filter(data_18.X > -122.41) \
                     .filter(data_18.X < -122.40) \
                     .filter(data_18.Y > 37.79) \
                     .filter(data_18.Y < 37.80) \
                     .filter(col('DayOfWeek')=='Sunday') \
                     .groupBy('Date').count().orderBy('Date') \
                     .toPandas())
  axes[i//2][i%2].plot(df[i]['Date'],df[i]['count'],'--.')
  axes[i//2][i%2].xaxis.set_major_formatter(monthsFmt)
  axes[i//2][i%2].set_title('{0}'.format(years[i]))
  axes[i//2][i%2].set_xlabel('month')
  axes[i//2][i%2].set_ylabel('# Crime on Sunday at Dt')
#   plt.setp(plt.xticks()[0], rotation=30, ha='right')
display()

# COMMAND ----------

# Old Method for Q3 (deprecated)
# import matplotlib.dates as mdates
# yearsFmt = mdates.DateFormatter('%m/%Y')

# fig, axes = plt.subplots(nrows=2,ncols=1)

# q3_17 = data_17.filter(data_17.X > -122.41).filter(data_17.X < -122.40).filter(data_17.Y > 37.79).filter(data_17.Y < 37.80)
# q3_17 = q3_17.filter(q3_17.DayOfWeek=='Sunday').groupBy('Date').count().toPandas()
# # q3_17['Date'] = pd.to_datetime(q3_17['Date']).dt.date
# q3_17 = q3_17.sort_values(by='Date')
# q3_17 = q3_17.reset_index(drop=True)
# q3_17 = q3_17.set_index('Date')
# axes[0].plot(q3_17['count'],'--.')
# axes[0].xaxis.set_major_formatter(yearsFmt)
# # plt.setp(plt.xticks()[0], rotation=30, ha='right')


# q3_18 = data_18.filter(data_18.X > -122.41).filter(data_18.X < -122.40).filter(data_18.Y > 37.79).filter(data_18.Y < 37.80)
# q3_18 = q3_18.filter(q3_18.DayOfWeek=='Sunday').groupBy('Date').count().toPandas()
# # q3_18['Date'] = pd.to_datetime(q3_18['Date']).dt.date
# q3_18 = q3_18.sort_values(by='Date')
# q3_18 = q3_18.set_index('Date')
# axes[1].plot(q3_18['count'],'--.')
# axes[1].xaxis.set_major_formatter(yearsFmt)
# plt.setp(plt.xticks()[1], rotation=30, ha='right')

# display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### OLAP
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

# DBTITLE 0,Data from 2015 to 2017
from pyspark.sql.functions import year, month
q4_1 = data_17.groupBy(year(col('Date')).alias('year'), month(col('Date')).alias('month')).count().orderBy('year','month').toPandas()
q4_2 = data_18.groupBy(year(col('Date')).alias('year'), month(col('Date')).alias('month')).count().orderBy('year','month').toPandas()
q4 = pd.concat([q4_1,q4_2],ignore_index=True)
q4 = q4.pivot(index='month', columns='year', values='count')
ax = q4.plot(kind='line')
# ax.legend(loc='best')
ax.set_title('Number of crimes in each month from 2015 - 2018')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### OLAP
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15, 2018/10/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

q5_15 = data_17.filter("Date == '2015-12-15'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2015').toPandas().set_index('hour')
q5_16 = data_17.filter("Date == '2016-12-15'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2016').toPandas().set_index('hour')
q5_17 = data_17.filter("Date == '2017-12-15'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2017').toPandas().set_index('hour')
q5_18 = data_18.filter("Date == '2018-10-15'").groupBy('hour').count().orderBy('hour').withColumnRenamed('count','2018').toPandas().set_index('hour')
q5 = pd.concat([q5_15, q5_16, q5_17, q5_18], axis=1)

ax = q5.plot(kind='bar',figsize=(16,10))
ax.set_title('Number of Crimes in day 12/15 (2015-2017) and 10/15(2018)')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### OLAP
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

from pyspark.sql.functions import upper, col
q6_17 = data_17.withColumnRenamed('PdDistrict','District').groupBy('District').count().withColumnRenamed('count','15_17').orderBy('15_17',ascending=False)
q6_18 = data_18.withColumn('District', upper(col('District'))).groupBy('District').count().withColumnRenamed('count','18').orderBy('18',ascending=False)
q6 = q6_17.join(q6_18,'District').withColumn('CrimeNum',col('15_17')+col('18')).drop('15_17','18')\
     .orderBy('CrimeNum',ascending=False).toPandas().set_index('District')
ax = q6.plot(kind='bar')
xticklabels = list(q6.index)
ax.set_xticklabels(xticklabels, rotation = 45, ha="center" ,fontsize=6)
ax.set_title('Total Number of Crime during 2015 - 2018 in each district ')
display()

# COMMAND ----------

q6_17 = data_17.select('*').groupBy('Category','hour').count().orderBy('hour').toPandas()
q6_17 = q6_17.pivot(index='Category', columns='hour', values='count')
q6_17 = q6_17.div(q6_17.sum(axis=0),axis=1)
fig, ax = plt.subplots(nrows=6,ncols=4,figsize=(20,20))
for i in q6_17.columns:
  labels = q6_17[i].sort_values(ascending=False)[:5].index.tolist()
  values = q6_17[i].sort_values(ascending=False)[:5].values
  labels.append('other')
  values = np.append(values, 1-values.sum())
  ax[i//4][i%4].pie(values, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
  ax[i//4][i%4].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  ax[i//4][i%4].set_title('Hour {}'.format(i))
fig.suptitle('Percentage of Category in 24 Hour from 2015-2017')
display()

# COMMAND ----------

q6_18 = data_18.select('*').groupBy('Category','hour').count().orderBy('hour').toPandas()
q6_18 = q6_18.pivot(index='Category', columns='hour', values='count')
q6_18 = q6_18.div(q6_18.sum(axis=0),axis=1)
fig, ax = plt.subplots(nrows=6,ncols=4,figsize=(20,20))
for i in q6_18.columns:
  labels = q6_18[i].sort_values(ascending=False)[:5].index.tolist()
  values = q6_18[i].sort_values(ascending=False)[:5].values
  labels.append('other')
  values = np.append(values, 1-values.sum())
  ax[i//4][i%4].pie(values, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
  ax[i//4][i%4].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  ax[i//4][i%4].set_title('Hour {}'.format(i))
fig.suptitle('Percentage of Category in 24 Hour in 2018')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### OLAP
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

q7_17_all = data_17.groupBy('Category').count().withColumnRenamed("count", "all").orderBy('all',ascending=False)
q7_17_null = data_17.filter(data_17['Resolution']!='NONE').groupBy('Category').count().withColumnRenamed("count", "resolved").orderBy('resolved',ascending=False)
q7_17 = q7_17_all.join(q7_17_null, 'Category').toPandas()
q7_17['Reso_rate'] = q7_17['resolved']/q7_17['all']
q7_17 = q7_17.set_index('Category')
q7_17 = q7_17.sort_values('Reso_rate',ascending=False)

# fig, ax = plt.subplots()
# q7_17.plot.bar(y='Reso_rate',figsize=(12,12))
# ax.set_xticklabels(q7_17.index,rotation=90,fontsize=4)
# ax.set_title('Resolution by category from 2015-2017')
# display()


fig, ax = plt.subplots(figsize=(12,12))
labels = q7_17.index.tolist()
ax.barh(np.arange(len(labels)), q7_17['Reso_rate'], align='center')
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=5)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Resolution Rate')
ax.set_title('Resolution by category during 2015 - 2017')
display()

# COMMAND ----------

q7_18_all = data_18.groupBy('Category').count().withColumnRenamed("count", "all").orderBy('all',ascending=False)
q7_18_null = data_18.filter(data_18['Resolution']!='Open or Active').groupBy('Category').count().withColumnRenamed("count", "resolved").orderBy('resolved',ascending=False)
q7_18 = q7_18_all.join(q7_18_null, 'Category').toPandas()
q7_18['Reso_rate'] = q7_18['resolved']/q7_18['all']
q7_18 = q7_18.set_index('Category')
q7_18 = q7_18.sort_values('Reso_rate',ascending=False)

fig, ax = plt.subplots(figsize=(12,12))
labels = q7_18.index.tolist()
ax.barh(np.arange(len(labels)), q7_18['Reso_rate'], align='center')
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=6)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Resolution Rate')
ax.set_title('Resolution by category in 2018')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Apply Spark ML clustering for spatial data analysis
# MAGIC Extra: visualize the spatial distribution of crimes and run a kmeans clustering algorithm (please use Spark ML kmeans)  
# MAGIC You can refer Spark ML Kmeans a example: https://spark.apache.org/docs/latest/ml-clustering.html#k-means

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

q8_17 = data_17.select('X','Y')
q8_18 = data_18.select('X','Y')
q8_train = q8_17.union(q8_18).groupBy('X','Y').count().orderBy('count',ascending=False).dropna()

vecAssembler = VectorAssembler(inputCols=["X", "Y"], outputCol="features")
kmeans = KMeans(k=5, seed=1)
pipeline = Pipeline(stages=[vecAssembler, kmeans])
model = pipeline.fit(q8_train)

# q8_res = q8_17.union(q8_18)
q8_res = model.transform(q8_train)
q8_res = q8_res.toPandas()

# COMMAND ----------

q8_res

# COMMAND ----------

fig, ax = plt.subplots()
ax.scatter(q8_res['X'], q8_res['Y'], c=(q8_res['prediction']),cmap=plt.cm.jet, alpha=0.9)
ax.set_title("5 Clsuters")
display()

# COMMAND ----------

kmeans = KMeans(k=8, seed=1)
pipeline = Pipeline(stages=[vecAssembler, kmeans])
model = pipeline.fit(q8_train)

# q8_res = q8_17.union(q8_18)
q8_res = model.transform(q8_train)
q8_res = q8_res.toPandas()

fig, ax = plt.subplots()
ax.scatter(q8_res['X'], q8_res['Y'], c=(q8_res['prediction']),cmap=plt.cm.jet, alpha=0.9)
ax.set_title("8 Clsuters")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###conclusion
# MAGIC I applied spark sql and spark dataframe to analyze the crime event data from San Francisco. The total number of crime events in the dataset is over 600,000 during year 2015 to 2018. From the data and visualization, here are some of the key findings: 
# MAGIC 1. Larceny/Theft is the most crime events happend in 2015 - 2018.
# MAGIC 2. The district 'Southern' , 'Northern', 'Mission' and 'Central' are the four most dangerous district. However, since 2018 public security in 'Central' as become worse and it has been the most dangerous place.
# MAGIC 3. Number of crime are more in months March, August, December. So we suggest visitors be careful when visiting San Francisco during these months. Especially in the downtown areas on Sundays.
# MAGIC 4. During time of a day, the number of crime are more at noon, late afternoon (17:00 ~ 18:00) and late night (22:00 - 00:00). Larceny/Theft are the most common events at all time, but in early morning around 5:00am, burglary event is more than other time.
# MAGIC 5. Events related to traffic, drug and warrants have higher resolution rate than others
