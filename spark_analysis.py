from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

from pymongo import MongoClient

def get_db():
    client = MongoClient("mongodb://localhost/")
    return client["car_insurance"]


spark = SparkSession.builder \
    .appName("Car Insurance Analysis") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost/car_insurance.analysis_results") \
    .getOrCreate()

# 2. Load CSV file
df = spark.read.csv("data/vehicles.csv", header=True, inferSchema=True)

# 3. Display schema info
print(df.columns)
df.printSchema()

# 4. Basic analysis results
print("\nAverage CREDIT_SCORE by INCOME group:")
avg_credit_by_income = df.groupBy("INCOME").avg("CREDIT_SCORE")
print("Data to be written to MongoDB:")
avg_credit_by_income.show()


print("\nAverage CREDIT_SCORE by EDUCATION level:")
df.groupBy("EDUCATION").avg("CREDIT_SCORE") \
    .withColumnRenamed("avg(CREDIT_SCORE)", "AVG_CREDIT_SCORE") \
    .write \
    .format("mongodb") \
    .option("collection", "avg_credit_by_education") \
    .mode("overwrite") \
    .save()


print("\nDriver count by AGE group:")
df.groupBy("AGE").count() \
    .withColumnRenamed("count", "DRIVER_COUNT") \
    .write \
    .format("mongodb") \
    .option("collection", "count_by_age_group") \
    .mode("overwrite") \
    .save()


print("\nAverage ANNUAL_MILEAGE by VEHICLE_TYPE:")
df.groupBy("VEHICLE_TYPE").avg("ANNUAL_MILEAGE") \
    .withColumnRenamed("avg(ANNUAL_MILEAGE)", "AVG_MILEAGE") \
    .write \
    .format("mongodb") \
    .option("collection", "avg_mileage_by_vehicle_type") \
    .mode("overwrite") \
    .save()


print("\nAverage OUTCOME by SPEEDING_VIOLATIONS:")
df.groupBy("SPEEDING_VIOLATIONS").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .write \
    .format("mongodb") \
    .option("collection", "avg_outcome_by_speeding") \
    .mode("overwrite") \
    .save()


print("\nAverage OUTCOME by DUIS:")
df.groupBy("DUIS").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .write \
    .format("mongodb") \
    .option("collection", "avg_outcome_by_duis") \
    .mode("overwrite") \
    .save()


# 5. Write result to MongoDB
avg_credit_by_income = avg_credit_by_income.withColumnRenamed("avg(CREDIT_SCORE)", "AVG_CREDIT_SCORE")

avg_credit_by_income.write \
    .format("mongodb") \
    .option("uri", "mongodb://localhost/") \
    .option("database", "car_insurance") \
    .option("collection", "analysis_results") \
    .mode("overwrite") \
    .save()

# Average OUTCOME by EDUCATION level
avg_outcome_by_education = df.groupBy("EDUCATION").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .dropna()

# Write to MongoDB
avg_outcome_by_education.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_outcome_by_education") \
    .save()


# Average OUTCOME by DUIS
avg_outcome_by_duis = df.groupBy("DUIS").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .dropna()

# Write to MongoDB
avg_outcome_by_duis.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_outcome_by_duis") \
    .save()

# Average OUTCOME by SPEEDING_VIOLATIONS
avg_outcome_by_speeding = df.groupBy("SPEEDING_VIOLATIONS").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .dropna()

# Write to MongoDB
avg_outcome_by_speeding.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_outcome_by_speeding") \
    .save()

# Average OUTCOME by AGE group
avg_outcome_by_age = df.groupBy("AGE").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .dropna()

# Write to MongoDB
avg_outcome_by_age.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_outcome_by_age") \
    .save()

df.groupBy("VEHICLE_TYPE").avg("OUTCOME") \
    .withColumnRenamed("avg(OUTCOME)", "AVG_OUTCOME") \
    .write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("collection", "analysis_outcome_by_vehicle_type") \
    .save()



vehicle_outcomes = df.groupBy("VEHICLE_TYPE").agg(avg("OUTCOME").alias("AVG_OUTCOME"))
vehicle_outcomes_pd = vehicle_outcomes.toPandas()
records = vehicle_outcomes_pd.to_dict(orient="records")
if records:
    collection = get_db()["vehicle_outcomes"]
    collection.delete_many({})
    collection.insert_many(records)


speeding_by_exp = df.groupBy("DRIVING_EXPERIENCE").agg(avg("SPEEDING_VIOLATIONS").alias("AVG_SPEEDING"))
speeding_by_exp_pd = speeding_by_exp.toPandas()
records = speeding_by_exp_pd.to_dict(orient="records")
if records:
    collection = get_db()["speeding_by_experience"]
    collection.delete_many({})
    collection.insert_many(records)


dui_outcomes = df.groupBy("DUIS").agg(avg("OUTCOME").alias("AVG_OUTCOME"))
dui_outcomes_pd = dui_outcomes.toPandas()
records = dui_outcomes_pd.to_dict(orient="records")
if records:
    collection = get_db()["dui_outcomes"]
    collection.delete_many({})
    collection.insert_many(records)


# 6. Show data samples
df.show(5)

# 7. Example: Average annual mileage
df.select("ANNUAL_MILEAGE").groupBy().avg().show()

# 8. Example: Count vehicles by type
df.groupBy("VEHICLE_TYPE").count().show()

# 9. Stop session
spark.stop()
