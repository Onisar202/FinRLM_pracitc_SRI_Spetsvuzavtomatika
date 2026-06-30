from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.finrlm.agent.tools import TOOLS

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            model="finrlm-llm",
            temperature=0.1
        )
        _agent = create_react_agent(llm, TOOLS)
    return _agent

def run_agent(query: str) -> str:
    agent = _get_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content