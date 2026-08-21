import os # --> imports the os module to interact with the operating system
import requests 
from dotenv import load_dotenv # --> imports the .env file and uses the load_dotenv function to load the variables into the environment
load_dotenv() # --> loads the variables into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

@tool('get_current_weather', description="Get the current weather info in a city", return_direct=False)
def get_current_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6),
    tools = [get_current_weather],
    system_prompt = "You are a weather assistant who always cracks jokes and is humorous while remaining helpful."
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather in California?'}
    ]
})

print(response["messages"][-1].content)
