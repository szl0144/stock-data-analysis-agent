# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM

# * Libraries

from langchain_openai import ChatOpenAI
import os
import yaml
import pandas as pd
from pprint import pprint

from ai_data_science_team.agents import data_cleaning_agent

# * Setup

MODEL    = "gpt-4o-mini"
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), "logs/")

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# 1.0 Data Cleaning Agent

data_cleaning_agent = data_cleaning_agent(model = llm, log=LOG, log_path=LOG_PATH)

df = pd.read_csv("data/churn_data.csv")

response = data_cleaning_agent.invoke({
    "user_instructions": "Don't remove outliers when cleaning the data.",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})

response.keys()

df

pd.DataFrame(response['data_cleaned'])

pprint(response['messages'][0].content)

