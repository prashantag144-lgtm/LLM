# DOCUMENTS --> LOADERS -->CHUNKING --> EMBEDDING --> VECTOR STORES --> PROMPTS -->RETRIEVERS


from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import RetrieverOutput
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

embedded=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store=Chroma.from_documents(
    embedding=embedded,
    documents=chunks,
    persist_directory="CHROMA_DB"
)

retriver=vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 5, 
        "fetch_k": 50
    }
)

system_prompt = """
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

prompt=ChatPromptTemplate.from_messages(
[
    ("system",system_prompt),
    ("human","""Retrieved SAP Documentation: {context}
    User Question:{question} Answer the question using the retrieved documentation."""),
    ]
)

llm=ChatGroq(model="openai/gpt-oss-120b",api_key="")
question=input("ENTER THE QUERY: ")

docs=retriver.invoke(question)

context = "\n\n".join(
        [doc.page_content for doc in docs]
)



final_prompt=prompt.invoke(
    {
    "context":context,
    "question":question
    }
)
print(context)
response=llm.invoke(final_prompt)
print(response.content)  