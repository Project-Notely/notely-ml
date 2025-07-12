import os
import urllib.parse

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv(dotenv_path=".env.local")

password = os.getenv("MONGO_PASSWORD")
encoded_password = urllib.parse.quote_plus(password)


uri = f"mongodb+srv://Isolumi:{encoded_password}@notely.a5h8v.mongodb.net/?retryWrites=true&w=majority&appName=Notely"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
