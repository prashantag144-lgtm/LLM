from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.messages import AIMessage,SystemMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser

llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")
query=input("ENTER THE QUERY")
prompt=([
    SystemMessage(content="You are a funny AI chatbot and also very entertaining chatbot"),
    HumanMessage(content=query)
])


response=llm.invoke(prompt)
output=AIMessage(content=response.content)
print(output)
