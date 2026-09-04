import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

file_path=input("ENTER YOUR FILE PATH : ")
file=Path(file_path)

loader=PyPDFLoader(file)
data=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    separators="",
    chunk_size=1000,
    chunk_overlap=150
)

chunks=text_splitter.split_documents(data)

embedded=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store=Chroma.from_documents(
    embedding=embedded,
    documents=chunks,
    persist_directory="CHROMADB"
)

retriver=vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs=
    {"k": 5, 
    "fetch_k": 50
    }
)

llm=ChatGroq(
    model="openai/gpt-oss-120b",api_key=""
)


lst=[]
while(True):
    query=input("ENTER A QUERY : ")

    if(query=="EIXT"):
        break
    else:
        

        system_message="""
        You are an expert SAP documentation analysis assistant.

        Your job is to answer questions using the retrieved SAP
        documentation provided by the application.

        Rules:

        1. Always prioritize the retrieved context.
        2. Never fabricate SAP information.
        3. If the context does not contain enough information,
        clearly state that.
        4. Preserve SAP terminology.
        5. If multiple pieces of context are relevant, combine them.
        6. If the context contains conflicting information,
        explicitly mention the conflict.
        7. When explaining procedures, use numbered steps.
        8. Mention document sections/pages when available.
        9. Answer only what the user asked.
        """


        prompt=ChatPromptTemplate.from_messages([
            ("system",system_message),
            ("human","""Retrieved SAP Documentation: {context}
            User Question:{question} Answer the question using the retrieved documentation."""),
        ])

        docs=retriver.invoke(query)

        for i in docs:
            lst.append(i.page_content)
            context="/n/n".join(lst)

        final_prompt=prompt.invoke(
            
                {"question":query,
                "context":context}
            
        )

        response=llm.invoke(final_prompt)
        print(response.content)

        


