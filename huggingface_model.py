import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint

load_dotenv()
api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not api_key: 
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")



llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)