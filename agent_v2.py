from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated


llm=ChatGroq(model="openai/gpt-oss-120b", api_key="")

def merge_scores_dictionaries(existing: dict, newupdate: dict):
    if existing is None:
        return newupdate

    return {**existing, **newupdate}


class parallelpipeline(TypedDict):
    job_description:str
    analyser_score:Annotated[dict[str,int],merge_scores_dictionaries]


def technical_analyser_node(state:parallelpipeline):
    
    prompt = (
                "Analyze the following job description from a technical perspective. "
                "Evaluate the quality and completeness of the technical requirements, "
                "including programming languages, frameworks, databases, cloud platforms, "
                "DevOps tools, and other relevant technical skills. "
                "Give a score from 0 to 100, where 0 means the job description has "
                "very poor or missing technical requirements, and 100 means it has "
                "excellent, clear, specific, and comprehensive technical requirements. "
                "Return ONLY the plain integer number, nothing else.\n\n"
                f"Job Description:\n{state['job_description']}"
            )
    response=llm.invoke(prompt)
    try:
        return {
                 "analyser_score":{"technical_score":int(response.content.strip())}
            }
    
    except Exception as e:
        print(e)
        


def qualitiy_score_node(state:parallelpipeline):
    prompt = (
    "Analyze the following job description for its overall quality and clarity. "
    "Evaluate whether the job description is clear, specific, well-structured, "
    "complete, professional, and easy for a candidate to understand. "
    "Consider whether the responsibilities, requirements, expectations, and "
    "other important information are sufficiently explained. "
    "Give a score from 0 to 100, where 0 means the job description is very "
    "poor, unclear, vague, or incomplete, and 100 means it is extremely clear, "
    "well-structured, specific, professional, and comprehensive. "
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Job Description:\n{state['job_description']}"
)
    response=llm.invoke(prompt)

    try:
        return{
            
                "analyser_score":{"quality_score":int(response.content.strip())}
          
        }
    except Exception as e:
        print(e)
        


def bias_score_node(state:parallelpipeline):
    prompt = (
    "Analyze the following job description for potential bias or discriminatory language. "
    "Look for gender bias, age bias, racial or cultural bias, disability-related bias, "
    "unfair exclusionary requirements, stereotypes, or wording that may discourage certain "
    "groups of qualified candidates from applying. "
    "Also consider whether the requirements unnecessarily favor or exclude a particular group. "
    "Give a score from 0 to 100, where 0 means the job description contains no significant "
    "bias or discriminatory language, and 100 means it contains highly concerning or "
    "discriminatory language. "
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Job Description:\n{state['job_description']}"
)
    response=llm.invoke(prompt)

    try:
        return{
            
                "analyser_score":{"bias_score":int(response.content.strip())}
            
        }
    except Exception as e:
        print(e)
        



graph=StateGraph(parallelpipeline)

graph.add_node("technical",technical_analyser_node)
graph.add_node("quality",qualitiy_score_node)
graph.add_node("bias",bias_score_node)

graph.add_edge(START,"technical")
graph.add_edge(START,"quality")
graph.add_edge(START,"bias")

graph.add_edge("technical",END)
graph.add_edge("quality",END)
graph.add_edge("bias",END)


app=graph.compile()

sample_job = """
We are looking for a Python developer to join our engineering team.

Requirements:
- 3+ years of Python experience
- Experience with FastAPI and Django
- Knowledge of PostgreSQL
- Experience with Docker and AWS
- Strong communication skills
- Must be energetic and able to work under extreme pressure
- Young and dynamic candidates preferred

Responsibilities:
- Build backend APIs
- Design database systems
- Deploy applications using Docker
- Work with the engineering team
"""

response=app.invoke({"job_description":sample_job})


print(response['analyser_score'])





