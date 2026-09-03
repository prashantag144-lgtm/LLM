from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage


llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")
memory=[]

memory.append(SystemMessage(content="You are a funny AI chatbot and also very entertaining chatbot"))

while(True):
    prompt=input("enter a query :")
    if (prompt=="EXIT"):
        break

    else:
        memory.append(HumanMessage(content=prompt))
        response=llm.invoke(memory)
        memory.append(AIMessage(content=response.content))
        print(response.content)







