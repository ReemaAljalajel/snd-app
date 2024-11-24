import os
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file

load_dotenv() 
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

print(LANGCHAIN_API_KEY)

load_dotenv(dotenv_path=".env", override=True)


LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
print(LANGCHAIN_API_KEY)