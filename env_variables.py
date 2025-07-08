import os
from dotenv import load_dotenv # type: ignore

#for API key in .env
load_dotenv()

api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not api_key: 
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")