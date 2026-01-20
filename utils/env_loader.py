import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, required: bool = True):
    value = os.getenv(key)
    if required and not value:
        raise ValueError(f"Missing environment variable: {key}")
    return value
