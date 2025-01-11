# TODO Create a SQLDataAnalyst class that inherits from DataAnalyst
# https://langchain-ai.github.io/langgraph/concepts/multi_agent/#multi-agent-architectures


# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain_openai import ChatOpenAI

# import os 
# import yaml


# MODEL    = "gpt-4o-mini"

# os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# llm = ChatOpenAI(model = MODEL)

# db = SQLDatabase.from_uri("sqlite:///data/northwind.db")

# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# tools = toolkit.get_tools()

# tools

# tools[0]
