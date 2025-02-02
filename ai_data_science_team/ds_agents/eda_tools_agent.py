

from typing import Any, Optional, Annotated, Sequence, List, Dict, Tuple
import operator
import pandas as pd
import os
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name

from ai_data_science_team.tools.eda import (
    describe_dataset, 
    visualize_missing, 
    correlation_funnel,
    generate_sweetviz_report,
)


AGENT_NAME = "exploratory_data_analyst_agent"

# Updated tool list for EDA
EDA_TOOLS = [
    describe_dataset,
    visualize_missing,
    correlation_funnel,
    generate_sweetviz_report,
]

class EDAToolsAgent(BaseAgent):
    """
    An Exploratory Data Analysis Tools Agent that interacts with EDA tools to generate summary statistics,
    missing data visualizations, correlation funnels, EDA reports, etc.
    
    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model for generating the tool-calling agent.
    create_react_agent_kwargs : dict
        Additional kwargs for create_react_agent.
    invoke_react_agent_kwargs : dict
        Additional kwargs for agent invocation.
    """
    
    def __init__(
        self, 
        model: Any,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        
    def _make_compiled_graph(self):
        """
        Creates the compiled state graph for the EDA agent.
        """
        self.response = None
        return make_eda_tools_agent(**self._params)
    
    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()
        
    async def ainvoke_agent(
        self, 
        user_instructions: str = None, 
        data_raw: pd.DataFrame = None, 
        **kwargs
    ):
        """
        Asynchronously runs the agent with user instructions and data.
        
        Parameters:
        ----------
        user_instructions : str, optional
            The instructions for the agent.
        data_raw : pd.DataFrame, optional
            The input data as a DataFrame.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs
        )
        self.response = response
        return None
    
    def invoke_agent(
        self, 
        user_instructions: str = None, 
        data_raw: pd.DataFrame = None, 
        **kwargs
    ):
        """
        Synchronously runs the agent with user instructions and data.
        
        Parameters:
        ----------
        user_instructions : str, optional
            The instructions for the agent.
        data_raw : pd.DataFrame, optional
            The input data as a DataFrame.
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs
        )
        self.response = response
        return None
    
    def get_internal_messages(self, markdown: bool = False):
        """
        Returns internal messages from the agent response.
        """
        pretty_print = "\n\n".join(
            [f"### {msg.type.upper()}\n\nID: {msg.id}\n\nContent:\n\n{msg.content}" 
             for msg in self.response["internal_messages"]]
        )
        if markdown:
            return Markdown(pretty_print)
        else:
            return self.response["internal_messages"]
    
    def get_artifacts(self, as_dataframe: bool = False):
        """
        Returns the EDA artifacts from the agent response.
        """
        if as_dataframe:
            return pd.DataFrame(self.response["eda_artifacts"])
        else:
            return self.response["eda_artifacts"]
    
    def get_ai_message(self, markdown: bool = False):
        """
        Returns the AI message from the agent response.
        """
        if markdown:
            return Markdown(self.response["messages"][0].content)
        else:
            return self.response["messages"][0].content

def make_eda_tools_agent(
    model: Any,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
):
    """
    Creates an Exploratory Data Analyst Agent that can interact with EDA tools.
    
    Parameters:
    ----------
    model : Any
        The language model used for tool-calling.
    create_react_agent_kwargs : dict
        Additional kwargs for create_react_agent.
    invoke_react_agent_kwargs : dict
        Additional kwargs for agent invocation.
    
    Returns:
    -------
    app : langgraph.graph.CompiledStateGraph
        The compiled state graph for the EDA agent.
    """
    
    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        eda_artifacts: dict

    def exploratory_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    * RUN REACT TOOL-CALLING AGENT FOR EDA")
        
        tool_node = ToolNode(
            tools=EDA_TOOLS
        )
        
        eda_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            **create_react_agent_kwargs,
        )
        
        response = eda_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
                "data_raw": state["data_raw"],
            },
            invoke_react_agent_kwargs,
        )
        
        print("    * POST-PROCESSING EDA RESULTS")
        
        internal_messages = response['messages']
        if not internal_messages:
            return {"internal_messages": [], "eda_artifacts": None}
        
        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)
        last_tool_artifact = None
        if len(internal_messages) > 1:
            last_message = internal_messages[-2]
            if hasattr(last_message, "artifact"):
                last_tool_artifact = last_message.artifact
            elif isinstance(last_message, dict) and "artifact" in last_message:
                last_tool_artifact = last_message["artifact"]
        
        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "eda_artifacts": last_tool_artifact,
        }
    
    workflow = StateGraph(GraphState)
    workflow.add_node("exploratory_agent", exploratory_agent)
    workflow.add_edge(START, "exploratory_agent")
    workflow.add_edge("exploratory_agent", END)
    
    app = workflow.compile()
    return app
