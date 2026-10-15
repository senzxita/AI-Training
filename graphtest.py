import os
from dotenv  import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6)

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}

graph_builder = StateGraph (MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile()


user_message = input('Enter a message: ')

print(graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}))


