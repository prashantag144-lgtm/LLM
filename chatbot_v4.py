from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")

while(True):
    query=input("ENTER A QUERY : ")
    if(query=="EXIT"):
        break
    else:
        prompt=[
        SystemMessage(content="You are a funny AI chatbot and also very entertaining chatbot"),
        HumanMessage(content=query)
    ]
    response=llm.invoke(prompt)
    result=AIMessage(content=response.content)
    print(result.content)




