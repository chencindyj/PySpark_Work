# Databricks notebook source
###Homework Assignment #3

# COMMAND ----------

#from pyspark.sql.functions import datediff, current_date, avg, round
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.sql import DataFrame

# COMMAND ----------

# MAGIC %md ##### Question 1: What was the average time each driver spent at the pit stop for each race?

# COMMAND ----------

#load data
df_pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header = True) 
df_drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header = True) 

# COMMAND ----------

#merge with driver names so we can identify the drivers
df_pitstops_drivers = df_drivers.select('driverId', 'forename', 'surname').join(df_pitstops, on = ['driverId'])

# COMMAND ----------

#aggregate data
df_pitstop_avgtime = df_pitstops_drivers.groupby('raceId', 'driverId', 'forename', 'surname').agg(avg('duration'))
df_pitstop_avgtime = df_pitstop_avgtime.withColumn('avg(duration)', round(df_pitstop_avgtime['avg(duration)'], 2)).withColumnRenamed('avg(duration)', 'Duration (secs)')

#display final result
display(df_pitstop_avgtime)

# COMMAND ----------

# MAGIC %md ##### Question 2: Rank the average time spent at the pit stop in order of who won each race

# COMMAND ----------

#load the standings
df_standings = spark.read.csv('s3://columbia-gr5069-main/raw/driver_standings.csv', header = True)

# COMMAND ----------

#join data
df_pitstop_standings = df_standings.select('driverId', 'raceId', 'position').join(df_pitstop_avgtime, on = ['driverId', 'raceId'])
display(df_pitstop_standings)

# COMMAND ----------

#since "position" is a string variable, convert to integer
df_pitstop_standings = df_pitstop_standings.withColumn('position', df_pitstop_standings['position'].cast(IntegerType()))

#sort by race and position
df_pitstop_standings = df_pitstop_standings.orderBy(asc('raceId'), asc('position'))

display(df_pitstop_standings)

# COMMAND ----------

# MAGIC %md ##### Question 3: Insert the missing code (e.g: ALO for Alonso) for drivers based on the 'drivers' dataset

# COMMAND ----------

#create a separate column that generates a three-letter code (using the first three characters from the surname)
df_drivers = df_drivers.withColumn('three_code', upper(df_drivers.surname.substr(1,3)))

#if the code is "\N" (equal to 2 character length), then we will replace the code with the new three-letter code
df_drivers = df_drivers.withColumn('code', when(length(df_drivers['code']) == 2, df_drivers['three_code']).otherwise(df_drivers['code']))

#display final result
display(df_drivers)

# COMMAND ----------

# MAGIC %md ##### Question 4: Who is the youngest and oldest driver for each race? Create a new column called “Age”

# COMMAND ----------

#add a column called age to the drivers data set
df_drivers = df_drivers.withColumn('age', datediff(current_date(),df_drivers.dob)/365.25)

# COMMAND ----------

#merge data with standings in order to match drivers to each race
df_race_age = df_standings.select('driverId', 'raceId').join(df_drivers, on = ['driverId'], how = 'left')

# COMMAND ----------

#create separate dataframes for minimum and maximum ages for each race, flag them as min_age or max_age
df_race_age_min = df_race_age.groupby('raceId').agg(min('age')).withColumnRenamed('min(age)', 'min_age')
df_race_age_max = df_race_age.groupby('raceId').agg(max('age')).withColumnRenamed('max(age)', 'max_age')

#combine the min and max data frames so there is a data frame that logs the min and max age for each race (though no name is currently attached)
df_race_age_minmax = df_race_age_min.join(df_race_age_max, on = ['raceId'])

# COMMAND ----------

display(df_race_age)

# COMMAND ----------

#merge new data frame with driver names and ages
df_race_age_combo = df_race_age_minmax.join(df_race_age, on = ['raceId'])

#create a new column that flags if someone is the youngest or oldest driver
df_race_age_combo = df_race_age_combo.withColumn('young_or_old', when(df_race_age_combo.age == df_race_age_combo.min_age, 'youngest').when(df_race_age_combo.age == df_race_age_combo.max_age, 'oldest'))

#combine the data frames and then sort by race
df_race_age_combo = df_race_age_combo.filter((df_race_age_combo.age == df_race_age_combo.min_age) | (df_race_age_combo.age == df_race_age_combo.max_age))
df_race_age_combo = df_race_age_combo.withColumn('age', df_drivers['age'].cast(IntegerType()))
df_race_age_combo = df_race_age_combo.select('raceId', 'driverId', 'code', 'forename', 'surname', 'dob', 'age', 'young_or_old').orderBy(asc('raceId'), asc('young_or_old'))

#display final result
display(df_race_age_combo)

# COMMAND ----------

# MAGIC %md ##### Question 5: For a given race, which driver has the most wins and losses?

# COMMAND ----------

#select certain columns from the previously joined data frame
df_win_loss = df_standings.select('driverId', 'raceId', 'position').join(df_drivers, on = ['driverId'], how = 'left')
df_win_loss = df_win_loss.select('driverId', 'raceId', 'code', 'driverRef', 'surname', 'position')
df_win_loss = df_win_loss.withColumn('driverId', df_win_loss['driverId'].cast(IntegerType()))

#create a column tallying wins and losses for each driver and race (only the driver who comes in first is considered the winner)
df_win_loss = df_win_loss.withColumn('wins', when(df_win_loss['position'] == 1, 1).otherwise(0))
df_win_loss = df_win_loss.withColumn('loss', when(df_win_loss['position'] != 1, 1).otherwise(0))

#aggregate the count of losses and wins for each driver
df_win_loss_summary = df_win_loss.groupby('driverId').agg(sum('wins'), sum('loss'))

#join each driver's win/loss record by race
df_win_loss_combo = df_win_loss.join(df_win_loss_summary, on = ['driverId'], how = 'left').select('driverId', 'surname', 'raceId', 'sum(wins)', 'sum(loss)').withColumnRenamed('sum(wins)', 'wins').withColumnRenamed('sum(loss)', 'losses')

# COMMAND ----------

#identify the most wins and losses by race, and create a separate data frame with this info
df_most_winloss = df_win_loss_combo.groupby('raceId').agg(max('wins'), max('losses'))

#join the win/loss summary to the original table
df_win_loss_combo = df_win_loss_combo.join(df_most_winloss, on = ['raceId']).withColumnRenamed('max(wins)', 'max_wins')
df_win_loss_combo = df_win_loss_combo.withColumnRenamed('max(losses)', 'max_losses')
df_win_loss_combo = df_win_loss_combo.filter((df_win_loss_combo.wins == df_win_loss_combo.max_wins) | (df_win_loss_combo.losses == df_win_loss_combo.max_losses))

# COMMAND ----------

#flag which driver in each race has the most wins/losses
df_win_loss_combo = df_win_loss_combo.withColumn('most_wins_losses', when(df_win_loss_combo.max_wins == df_win_loss_combo.wins, 'most wins').when(df_win_loss_combo.max_losses == df_win_loss_combo.losses, 'most losses')).select('raceId', 'driverId', 'surname', 'wins', 'losses', 'most_wins_losses').orderBy(asc('raceId'), asc('most_wins_losses'))

#display final result
display(df_win_loss_combo)

# COMMAND ----------

# MAGIC %md ##### Question 6: Continue exploring the data by answering your own question.
# MAGIC ##### My question will be: which constructor has been the most successful in the new Millennium (2000 and beyond)?

# COMMAND ----------

#load data
constructor_results = spark.read.csv('s3://columbia-gr5069-main/raw/constructor_results.csv', header = True)
constructor_names = spark.read.csv('s3://columbia-gr5069-main/raw/constructors.csv', header = True)
races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header = True)

#join data on constructor ID
constructor_races = constructor_results.join(races, on = 'raceId')
constructor_races = constructor_races.filter(constructor_races.year >= 2000).select('raceId', 'constructorId', 'points')
display(constructor_races)

# COMMAND ----------

#join constructor descriptions to the last 10 years' results
df_constructor = constructor_names.select('constructorId', 'name', 'nationality').join(constructor_races, on = ['constructorId'])

#aggregate based on points (since )
df_constructor_wins = df_constructor.groupby('constructorId', 'name', 'nationality').agg(sum('points')).orderBy(desc('sum(points)')).withColumnRenamed('sum(points)', 'total_points')

#display final result
display(df_constructor_wins)
