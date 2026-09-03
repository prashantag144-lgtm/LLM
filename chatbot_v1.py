from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")

message=input("ENTER A QUESTION : ")
response=llm.invoke(message)
print(response.content)
