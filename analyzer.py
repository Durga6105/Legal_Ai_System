import asyncio
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import get_llm

llm = get_llm()

class AgentState(TypedDict, total=False):
    document_text: str
    chunks: List[str]
    output: str

def chunk_node(state):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return {"chunks": splitter.split_text(state["document_text"])}

async def analyze_chunk(chunk):
    prompt = f"""
    You are a legal expert.
    Analyze this NDA and check:
    - Confidentiality
    - Definition of confidential information
    - Exclusions
    - Termination
    - Obligations
    - Liability
    - Jurisdiction
    - Data protection
    Return:
    Clause | Status (satisfied/missing/risky) | Reason
    TEXT:
    {chunk}
    """
    res = await llm.ainvoke(prompt)
    return res.content

async def analysis_async(state):
    tasks = [analyze_chunk(c) for c in state["chunks"]]
    results = await asyncio.gather(*tasks)
    return {"output": "\n".join(results)}

def analysis_node(state):
    return asyncio.run(analysis_async(state))

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("chunk", chunk_node)
    builder.add_node("analysis", analysis_node)
    builder.add_edge(START, "chunk")
    builder.add_edge("chunk", "analysis")
    builder.add_edge("analysis", END)
    return builder.compile()