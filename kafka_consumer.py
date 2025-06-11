from kafka import KafkaConsumer
import json
from pymongo import MongoClient

# MongoDB setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["car_insurance"]
collection = db["kafka_predictions"]

# Kafka consumer setup
consumer = KafkaConsumer(
    'insurance_predictions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='insurance-consumer-group'
)

print("👂 Listening to Kafka topic: 'insurance_predictions'...\n")

for message in consumer:
    data = message.value
    print("📥 New Insurance Prediction:")
    print(json.dumps(data, indent=2))

    # ✅ Save to MongoDB
    collection.insert_one(data)
    print("✅ Saved to MongoDB\n")
