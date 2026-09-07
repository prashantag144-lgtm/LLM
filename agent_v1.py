"""
Build an AI pipeline that takes a raw news article and produces:

Cleaned article
Short summary
Important keywords
Social-media post
"""

# 4 Nodes  

# Cleaner -->Summariser --> Keyword Extractor --> Social Media Writer

from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph import START,END


llm=ChatGroq(
    model="openai/gpt-oss-120b", api_key=""
)

# CREATING STATE

class articepipeline(TypedDict):
    raw_article:str
    cleaned_article:str
    summary:str
    keywords:str
    social_post:str
    

# CREATING NODES 


def cleaned_node(state:articepipeline)->dict:
    """Stage 1: Cleans and refines the raw_input or text here in this node"""


    prompt = (
    "You are an expert news editor and copyeditor. "
    "Clean and refine the following raw news article. "
    "Fix grammar, spelling, punctuation, sentence structure, and awkward wording. "
    "Improve readability and flow while preserving the original meaning, facts, "
    "names, numbers, dates, and important details. "
    "Do not add new information, opinions, assumptions, or conclusions. "
    "Do not summarize the article. "
    "Return only the cleaned and polished article.\n\n"
    f"Raw Article:\n{state['raw_article']}"
)
    response=llm.invoke(prompt)

    return{
        "cleaned_article":response.content.strip()
    }


def summary_node(state:articepipeline)->dict:

    prompt = (
    "You are an expert news summarizer. "
    "Read the following cleaned news article and create a concise, accurate summary. "
    "Include only the most important information such as the main event, key people or organizations, "
    "important facts, numbers, dates, causes, and outcomes. "
    "Do not add any information that is not present in the article. "
    "Do not give your opinion. "
    "Keep the summary clear, factual, and easy to understand. "
    "Return only the summary.\n\n"
    f"Cleaned Article:\n{state['cleaned_article']}"
)

    response=llm.invoke(prompt)

    return {"summary":response.content.strip()}

def keyword_node(state:articepipeline)->dict:

    prompt = (
    "You are an expert news analyst. "
    "Extract the most important keywords and key phrases from the following news article. "
    "Focus on people, organizations, places, events, topics, technologies, products, and important concepts. "
    "Avoid common or unnecessary words. "
    "Return only the keywords as a comma-separated list, without explanations.\n\n"
    f"Article:\n{state['cleaned_article']}"
)

    response=llm.invoke(prompt)

    return{
        "keywords":response.content.strip()
    }



def socialpost_node(state:articepipeline)->dict:

    prompt = (
    "You are an expert social media content creator. "
    "Convert the following news article into an engaging social media post. "
    "Capture the main news and most important facts accurately. "
    "Make the post concise, attention-grabbing, and easy to read. "
    "Use a strong opening hook and a natural conversational tone. "
    "Do not add information, opinions, or facts that are not present in the article. "
    "Do not exaggerate or misrepresent the news. "
    "Return only the final social media post.\n\n"
    f"Article:\n{state['cleaned_article']}"
)
    response=llm.invoke(prompt)

    return {
        "social_post":response.content.strip()
    }


graph=StateGraph(articepipeline)

graph.add_node("cleaner",cleaned_node)
graph.add_node("summary",summary_node)
graph.add_node("keyword",keyword_node)
graph.add_node("post",socialpost_node)

graph.add_edge(START, "cleaner")
graph.add_edge("cleaner", "summary")
graph.add_edge("summary", "keyword")
graph.add_edge("keyword", "post")
graph.add_edge("post", END)

app=graph.compile()

article=article = """
A new artificial intelligence platform has been launched to help small businesses
automate routine tasks such as customer support, data analysis, and appointment
scheduling. The platform uses AI agents that can understand user requests,
plan multiple steps, and interact with external tools.

According to the company, the system is designed to reduce the amount of time
employees spend on repetitive tasks while allowing them to focus on more complex
work. The platform also includes monitoring features that allow businesses
to review the actions taken by AI agents.

The company said the platform will initially be available to small and
medium-sized businesses and will later be expanded to larger organizations.
Industry experts believe that AI agent technology could significantly change
how businesses use software, although concerns about reliability, security,
and human oversight remain.
"""

result=app.invoke({
    "raw_article":article
})

print(result["social_post"])