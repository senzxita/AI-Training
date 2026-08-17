import os # --> imports the os module to interact with the operating system
from dotenv import load_dotenv # --> imports the .env file and uses the load_dotenv function to load the variables into the environment
load_dotenv() # --> loads the variables into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(
    model="gpt-4.1-mini", # --> model is the model to use for the LLM
    temperature=0.6) # --> temperature is the degree of randomness of the model's output

# --------- Using SimpleSequentialChain ---------

# Chain 1: Suggest a name for the AI assistant
name_prompt = PromptTemplate(
    template="I want you to suggest a single {culture_name} name for my new AI assistant",
    input_variables=["culture_name"]
) 

name_chain = name_prompt | llm
name = name_chain.invoke(
    {"culture_name": "Thai"}
).content.strip()

# Chain 2: Suggest a greeting based on the name selected for the AI assistant
greeting_prompt = PromptTemplate(
    template="""
Your name is {name}.

You are an AI assistant.

Introduce yourself in the {culture_name} language.

Your introduction should:
- Mention your name.
- Greet the user in the native language.
- Briefly explain that you are an AI assistant.
- Do not translate into English.
""",
    input_variables=["name", "culture_name"]
)
greeting_chain = greeting_prompt | llm
response = greeting_chain.invoke({"name": name, "culture_name": "Thai"})
print (response.content)



