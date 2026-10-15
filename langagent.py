import os
import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Fixed model name
llm = ChatOpenAI(model="gpt-4.1-mini")

# Schema for structured output
class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(
        ..., description='Classify whether the user wants to chat, ask for knowledge, or change code.'
    )

# Custom Graph State with agent tracking
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: Literal['chat', 'knowledge', 'code'] | None
    current_agent: str | None  # Tracks which agent answered the previous turn

# 1. Classification Node
def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    
    classification_sys_msg = SystemMessage(
        content='Determine whether the user wants casual chat ("chat"), knowledge retrieval ("knowledge"), or code editing ("code").'
    )
    
    result: IntentClassifier = structured_llm.invoke([classification_sys_msg] + state['messages'])
    return {'message_intent': result.message_intent}

# 2. Agent Nodes
def prompt_llm_chat(state: State):
    prev_agent = state.get('current_agent')
    intro = "" if prev_agent == 'chat' else "I am the Chat Agent. "
    
    sys_prompt = SystemMessage(
        content=f'You are a conversational chatbot. {f"Start your reply with: \'{intro}\'" if intro else ""} Answer the user.'
    )
    response = llm.invoke([sys_prompt] + state['messages'])
    return {'messages': [response], 'current_agent': 'chat'}

def prompt_llm_rag(state: State):
    prev_agent = state.get('current_agent')
    intro = "" if prev_agent == 'knowledge' else "I am the RAG Agent. "
    
    sys_prompt = SystemMessage(
        content=f'You are a Knowledge/RAG agent. {f"Start your response with \'{intro}\'" if intro else ""} Answer the user question accurately based on facts.'
    )
    response = llm.invoke([sys_prompt] + state['messages'])
    return {'messages': [response], 'current_agent': 'knowledge'}

def prompt_llm_code(state: State):
    prev_agent = state.get('current_agent')
    intro = "" if prev_agent == 'code' else "I am the Coding Agent. "
    
    sys_prompt = SystemMessage(
        content=f'You are a Coding Assistant. {f"Start your response with \'{intro}\'" if intro else ""} Help write or edit code.'
    )
    response = llm.invoke([sys_prompt] + state['messages'])
    return {'messages': [response], 'current_agent': 'code'}

# Router Logic
def route_intent(state: State) -> Literal['chat_agent', 'rag_agent', 'coding_agent']:
    intent = state.get('message_intent')
    if intent == 'knowledge':
        return 'rag_agent'
    elif intent == 'code':
        return 'coding_agent'
    return 'chat_agent'

# 3. Graph Assembly
graph_builder = StateGraph(State) 

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')

graph_builder.add_conditional_edges(
    'classifier',
    route_intent,
    {
        'chat_agent': 'chat_agent',
        'rag_agent': 'rag_agent',
        'coding_agent': 'coding_agent'
    }
)

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Fixed string UUID conversion
config = {'configurable': {'thread_id': str(uuid.uuid4())}}

# 4. Interactive Execution Loop
config = {'configurable': {'thread_id': str(uuid.uuid4())}}

if __name__ == "__main__":
    print("Router active. Type 'exit' to quit.\n")
    while True:
        user_message = input('Enter message: ')
        if user_message.lower() in ['exit', 'quit']:
            break

        if not user_message.strip():
            continue  # Prevent empty inputs from triggering LLM calls

        result = graph.invoke(
            {'messages': [HumanMessage(content=user_message)]}, 
            config=config
        )
        
        last_response = result['messages'][-1]
        intent = result.get('message_intent', 'UNKNOWN').upper()
        
        print(f"\n[{intent}] {last_response.content}\n")
    