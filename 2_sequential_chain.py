from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
# If we want to create new project in langsmith we can change this variable via env file or here by accessing os and changing the LANGCHAIN_PROJECT value

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM APP'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# If we want to add or make our own metadata we will use config variable which is in Dictionary format. So can see it in langsmith.
config = {
    # I can also change the runnable_sequence to my preferred name
    "run_name": "Unemployment Report",
    "tags" : ["report", "summary"],
    "metadata" : {"topic" : "Unemployment in India", "parser": "StrOutputParser", "model": "gemini-2.5-flash", "temperature": 0.7}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
