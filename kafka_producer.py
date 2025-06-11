from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_to_kafka(topic, data):
    try:
        producer.send(topic, value=data)
        producer.flush()
        print(f"✅ Sent to Kafka topic '{topic}':", data)
    except Exception as e:
        print("❌ Kafka send failed:", e)

# Example usage:
if __name__ == '__main__':
    test_data = {
        "age": 30,
        "credit_score": 0.7,
        "prediction": 28500.0
    }
    send_to_kafka("insurance_predictions", test_data)
