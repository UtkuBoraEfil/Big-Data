from pymongo import MongoClient

def get_collection(name="analysis_results"):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["car_insurance"]
    return db[name]

