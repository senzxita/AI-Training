import os
import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Fixed model name
llm = ChatOpenAI(model="gpt-4.1-mini")

# Schema for structured output
class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(
        ..., description='Classify whether the user wants to chat, ask for knowledge, or change code.'
    )

# Custom State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

# 1. Classification Node
def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    last_msg = state['messages'][-1]
    user_text = last_msg.content if hasattr(last_msg, 'content') else last_msg['content']

    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code").'},
        {'role': 'user', 'content': user_text}
    ])

    return {'message_intent': result.message_intent}

# 2. Agent Nodes
def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def prompt_llm_rag(state: State):
    messages = [{'role': 'system', 'content': 'No matter what the user says, always say "I am the RAG agent".'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def prompt_llm_code(state: State):
    messages = [{'role': 'system', 'content': 'No matter what the user says, always say "I am the CODING agent".'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# 3. Graph Assembly
graph_builder = StateGraph(State) 

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges(
    'classifier',
    lambda state: state['message_intent'],
    {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'coding_agent'}
)
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Fixed string UUID conversion
config = {'configurable': {'thread_id': str(uuid.uuid4())}}

# 4. Interactive Execution Loop
while True:
    user_message = input('Enter message: ')
    if user_message.lower() in ['exit', 'quit']:
        break

    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)
    
    last_response = result['messages'][-1]
    output_text = last_response.content if hasattr(last_response, 'content') else last_response['content']
    print(f"[{result['message_intent'].upper()}] {output_text}")
    