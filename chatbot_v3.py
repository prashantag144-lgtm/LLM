from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate


llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")
query=input("ENTER A QUERY : ")
prompt=ChatPromptTemplate.from_messages(
    [("system","You are a funny AI chatbot and also very entertaining chatbot"),
    ("human","{query}")]
)

message=prompt.invoke({"query":query})

response=llm.invoke(message)
print(response.content)
