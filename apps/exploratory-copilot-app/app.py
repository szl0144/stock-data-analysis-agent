# BUSINESS SCIENCE
# Exploratory Data Analysis (EDA) Copilot App
# -----------------------

# This app helps you search for data and produces exploratory analysis reports.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import html  

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict

# =============================================================================
# STREAMLIT APP SETUP (including data upload, API key, etc.)
# =============================================================================

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']
TITLE = "Your Exploratory Data Analysis (EDA) Copilot"
st.set_page_config(page_title=TITLE, page_icon="📊")
st.title("📊 " + TITLE)

st.markdown("""
Welcome to the EDA Copilot. This AI agent is designed to help you find and load data 
and return exploratory data analysis reports that can be used to understand the data 
prior to other analysis (e.g. modeling, feature engineering, etc).
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tools do you have access to? Return a table.
        - Give me information on the correlation funnel tool.
        - Explain the dataset.
        - What do the first 5 rows contain?
        - Describe the dataset.
        - Analyze missing data in the dataset.
        - Generate a correlation funnel. Use the Churn feature as the target.
        - Generate a Sweetviz report for the dataset. Use the Churn feature as the target.
        """
    )

# Sidebar for file upload / demo data
st.sidebar.header("EDA Copilot: Data Upload/Selection", divider=True)
st.sidebar.header("Upload Data (CSV or Excel)")
use_demo_data = st.sidebar.checkbox("Use demo data", value=False)

if "DATA_RAW" not in st.session_state:
    st.session_state["DATA_RAW"] = None

if use_demo_data:
    demo_file_path = Path("data/churn_data.csv")
    if demo_file_path.exists():
        df = pd.read_csv(demo_file_path)
        file_name = "churn_data"
        st.session_state["DATA_RAW"] = df.copy()
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.error(f"Demo data file not found at {demo_file_path}. Please ensure it exists.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.session_state["DATA_RAW"] = df.copy()
        file_name = Path(uploaded_file.name).stem
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.info("Please upload a CSV or Excel file or Use Demo Data to proceed.")

# Sidebar: OpenAI API Key and Model Selection
st.sidebar.header("Enter your OpenAI API Key")
st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key", 
    type="password", 
    help="Your OpenAI API key is required for the app to function."
)

if st.session_state["OPENAI_API_KEY"]:
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
    try:
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()

model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
OPENAI_LLM = ChatOpenAI(
    model=model_option,
    api_key=st.session_state["OPENAI_API_KEY"]
)
llm = OPENAI_LLM

# =============================================================================
# CHAT MESSAGE HISTORY AND STORAGE FOR PLOTS/DATAFRAMES
# =============================================================================

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index])
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)

display_chat_history()

# =============================================================================
# PROCESS AGENTS AND ARTIFACTS
# =============================================================================

def process_exploratory(question: str, llm, data: pd.DataFrame) -> dict:
    """
    This function initializes and calls the EDA agent using the provided question and data.
    It inspects the returned tool calls and artifacts, and then processes the artifact based on
    the tool that was called.
    
    Returns a dictionary containing the AI's text response plus any processed artifacts.
    """
    # Initialize the agent with a recursion limit for multi-step tasks
    eda_agent = EDAToolsAgent(
        llm, 
        invoke_react_agent_kwargs={"recursion_limit": 10},
    )
    
    # Invoke the agent using the user's question and the provided data
    eda_agent.invoke_agent(
        user_instructions=question,
        data_raw=data,
    )
    
    # Retrieve outputs from the agent
    tool_calls = eda_agent.get_tool_calls()
    ai_message = eda_agent.get_ai_message(markdown=False)
    artifacts = eda_agent.get_artifacts(as_dataframe=False)
    
    # Start building the result dictionary
    result = {
        "ai_message": ai_message,
        "tool_calls": tool_calls,
        "artifacts": artifacts
    }
    
    # If any tool was called, capture the last tool call details
    if tool_calls:
        last_tool_call = tool_calls[-1]
        result["last_tool_call"] = last_tool_call
        tool_name = last_tool_call
        
        print(f"Tool Name: {tool_name}")
        
        # Dispatch artifact processing based on the tool name
        if tool_name == "explain_data":
            # The explain_data tool returns a text explanation only.
            result["explanation"] = ai_message
            
        elif tool_name == "describe_dataset":
            # The describe_dataset tool returns a text summary and an artifact with key "describe_df"
            if artifacts and isinstance(artifacts, dict) and "describe_df" in artifacts:
                try:
                    df = pd.DataFrame(artifacts["describe_df"])
                    result["describe_df"] = df
                except Exception as e:
                    st.error(f"Error processing describe_dataset artifact: {e}")
                    
        elif tool_name == "visualize_missing":
            # The visualize_missing tool returns several base64-encoded images (one for each plot)
            if artifacts and isinstance(artifacts, dict):
                try:
                    # Process each missing data plot (matrix, bar, heatmap)
                    matrix_fig = matplotlib_from_base64(artifacts.get("matrix_plot"))
                    bar_fig    = matplotlib_from_base64(artifacts.get("bar_plot"))
                    heatmap_fig= matplotlib_from_base64(artifacts.get("heatmap_plot"))
                    result["matrix_plot_fig"] = matrix_fig
                    result["bar_plot_fig"] = bar_fig
                    result["heatmap_plot_fig"] = heatmap_fig
                except Exception as e:
                    st.error(f"Error processing visualize_missing artifact: {e}")
                    
        elif tool_name == "correlation_funnel":
            # The correlation_funnel tool returns correlation data, a static plot image, and a Plotly figure
            if artifacts and isinstance(artifacts, dict):
                if "correlation_data" in artifacts:
                    try:
                        corr_df = pd.DataFrame(artifacts["correlation_data"])
                        result["correlation_data"] = corr_df
                    except Exception as e:
                        st.error(f"Error processing correlation_data: {e}")
                if "plotly_figure" in artifacts:
                    try:
                        corr_plotly = plotly_from_dict(artifacts["plotly_figure"])
                        result["correlation_plotly"] = corr_plotly
                    except Exception as e:
                        st.error(f"Error processing correlation funnel Plotly figure: {e}")
                    
        elif tool_name == "generate_sweetviz_report":
            # The generate_sweetviz_report tool returns a file path (and possibly the HTML content) for the report
            if artifacts and isinstance(artifacts, dict):
                result["report_file"] = artifacts.get("report_file")
                result["report_html"] = artifacts.get("report_html")
                
        else:
            # Fallback for unrecognized tool calls: try to process common artifact keys
            if artifacts and isinstance(artifacts, dict):
                if "plotly_figure" in artifacts:
                    try:
                        plotly_fig = plotly_from_dict(artifacts["plotly_figure"])
                        result["plotly_fig"] = plotly_fig
                    except Exception as e:
                        st.error(f"Error processing Plotly figure: {e}")
                if "plot_image" in artifacts:
                    try:
                        fig = matplotlib_from_base64(artifacts["plot_image"])
                        result["matplotlib_fig"] = fig
                    except Exception as e:
                        st.error(f"Error processing matplotlib image: {e}")
                if "dataframe" in artifacts:
                    try:
                        df = pd.DataFrame(artifacts["dataframe"])
                        result["dataframe"] = df
                    except Exception as e:
                        st.error(f"Error converting artifact to dataframe: {e}")
    else:
        # No tool was called, so return the plain text response
        result["plain_response"] = ai_message
        
    return result

# =============================================================================
# MAIN INTERACTION: GET USER QUESTION AND HANDLE RESPONSE
# =============================================================================

if st.session_state["DATA_RAW"] is not None and (question := st.chat_input("Enter your question here:", key="query_input")):
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()
    
    with st.spinner("Thinking..."):
        # Add the user's message to the chat history
        st.chat_message("human").write(question)
        msgs.add_user_message(question)
        
        try:
            # Call the updated process_exploratory function
            result = process_exploratory(
                question, 
                llm, 
                st.session_state["DATA_RAW"]
            )
        except Exception as e:
            error_text = f"Sorry, I'm having difficulty processing your question. Error: {e}"
            msgs.add_ai_message(error_text)
            st.chat_message("ai").write(error_text)
            st.error(e)
            st.stop()
        
        # First, display the AI text response
        ai_text = result.get("ai_message", "")
        msgs.add_ai_message(ai_text)
        st.chat_message("ai").write(ai_text)
        
        # print(result)
        print(result.keys())

        # --- Display Content by Tool ---
        # Check if a tool was called
        if "last_tool_call" in result:
            tool_name = result["last_tool_call"]

            st.info(f"Tool used: **{tool_name}**")
            
            # Display based on the tool name
            if tool_name == "explain_data":
                # Display the explanation text from the explain_data tool.
                # if "explanation" in result:
                #     with st.expander("Data Explanation"):
                #         st.write(result["explanation"])
                pass
                        
            elif tool_name == "describe_dataset":
                # Display the description dataframe returned by the describe_dataset tool.
                if "describe_df" in result:
                    with st.expander("Dataset Description"):
                        st.dataframe(result["describe_df"])
                        
            elif tool_name == "visualize_missing":
                # Display the missing data plots.
                with st.expander("Missing Data Visualizations"):
                    if "matrix_plot_fig" in result:
                        st.subheader("Missing Data Matrix")
                        fig = result["matrix_plot_fig"]
                        # If the figure is a tuple, extract the first element.
                        if isinstance(fig, tuple):
                            fig = fig[0]
                        st.pyplot(fig)
                    if "bar_plot_fig" in result:
                        st.subheader("Missing Data Bar Plot")
                        fig = result["bar_plot_fig"]
                        if isinstance(fig, tuple):
                            fig = fig[0]
                        st.pyplot(fig)
                    if "heatmap_fig" in result:
                        st.subheader("Missing Data Heatmap")
                        fig = result["heatmap_plot_fig"]
                        if isinstance(fig, tuple):
                            fig = fig[0]
                        st.pyplot(fig)
                        
            elif tool_name == "correlation_funnel":
                # Display correlation data and plots.
                with st.expander("Correlation Funnel Analysis"):
                    if "correlation_data" in result:
                        st.subheader("Correlation Data")
                        st.dataframe(result["correlation_data"])
                    if "correlation_plot" in result:
                        st.subheader("Correlation Funnel (Static Plot)")
                        st.pyplot(result["correlation_plot"])
                    if "correlation_plotly" in result:
                        st.subheader("Correlation Funnel (Interactive Plotly)")
                        st.plotly_chart(result["correlation_plotly"])
                        
            elif tool_name == "generate_sweetviz_report":
                with st.expander("Sweetviz Report"):
                    # Read the report file
                    report_path = result["report_file"]
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_html = f.read()
                    except Exception as e:
                        st.error(f"Could not open report file: {e}")
                        report_html = "<h1>Report not found</h1>"

                    # Escape the report HTML so it can be safely embedded in the srcdoc attribute.
                    report_html_escaped = html.escape(report_html, quote=True)

                    # Build the HTML that embeds the report in an iframe with a full-screen toggle.
                    html_code = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <meta charset="utf-8">
                    <title>Sweetviz Report</title>
                    <style>
                        body, html {{
                        margin: 0;
                        padding: 0;
                        height: 100%;
                        }}
                        #iframe-container {{
                        position: relative;
                        width: 100%;
                        height: 600px;
                        }}
                        #myIframe {{
                        width: 100%;
                        height: 100%;
                        border: none;
                        }}
                        #fullscreen-btn {{
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        z-index: 1000;
                        padding: 8px 12px;
                        background-color: #007bff;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        }}
                    </style>
                    </head>
                    <body>
                    <div id="iframe-container">
                        <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
                        <iframe id="myIframe" srcdoc="{report_html_escaped}" allowfullscreen></iframe>
                    </div>
                    <script>
                        function toggleFullscreen() {{
                        var container = document.getElementById("iframe-container");
                        if (!document.fullscreenElement) {{
                            container.requestFullscreen().catch(err => {{
                            alert("Error attempting to enable full-screen mode: " + err.message);
                            }});
                            document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
                        }} else {{
                            document.exitFullscreen();
                            document.getElementById("fullscreen-btn").innerText = "Full Screen";
                        }}
                        }}
                        
                        // Listen for fullscreen change events to update button text if the user exits full-screen via ESC.
                        document.addEventListener('fullscreenchange', () => {{
                        if (!document.fullscreenElement) {{
                            document.getElementById("fullscreen-btn").innerText = "Full Screen";
                        }}
                        }});
                    </script>
                    </body>
                    </html>
                    """
                    # Render the HTML component. Adjust the height as needed.
                    components.html(html_code, height=620)
                        
            else:
                # Fallback: if the tool name is not explicitly handled, try common artifact keys.
                with st.expander("Additional Artifacts"):
                    if "plotly_fig" in result:
                        st.plotly_chart(result["plotly_fig"])
                    if "matplotlib_fig" in result:
                        st.pyplot(result["matplotlib_fig"])
                    if "dataframe" in result:
                        st.dataframe(result["dataframe"])
        else:
            # No tool was called; display a plain text response if available.
            # if "plain_response" in result:
            #     st.write(result["plain_response"])
            pass




# html_code = """
# <!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8">
#   <title>Iframe Full-Screen Toggle</title>
#   <style>
#     body, html {
#       margin: 0;
#       padding: 0;
#       height: 100%;
#     }
#     #iframe-container {
#       position: relative;
#       width: 100%;
#       height: 600px;
#     }
#     #myIframe {
#       width: 100%;
#       height: 100%;
#       border: none;
#     }
#     #fullscreen-btn {
#       position: absolute;
#       top: 10px;
#       right: 10px;
#       z-index: 1000;
#       padding: 8px 12px;
#       background-color: #007bff;
#       color: white;
#       border: none;
#       border-radius: 4px;
#       cursor: pointer;
#     }
#   </style>
# </head>
# <body>
#   <div id="iframe-container">
#     <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
#     <iframe id="myIframe" src="https://www.wikipedia.org" allowfullscreen></iframe>
#   </div>
#   <script>
#     function toggleFullscreen() {
#       var container = document.getElementById("iframe-container");
#       if (!document.fullscreenElement) {
#         container.requestFullscreen().catch(err => {
#           alert("Error attempting to enable full-screen mode: " + err.message);
#         });
#         document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
#       } else {
#         document.exitFullscreen();
#         document.getElementById("fullscreen-btn").innerText = "Full Screen";
#       }
#     }
    
#     // Optional: Listen for fullscreen change events to update button text if the user exits full-screen via ESC.
#     document.addEventListener('fullscreenchange', () => {
#       if (!document.fullscreenElement) {
#         document.getElementById("fullscreen-btn").innerText = "Full Screen";
#       }
#     });
#   </script>
# </body>
# </html>
# """

# # Render the HTML component in Streamlit. Adjust the height if necessary.
# components.html(html_code, height=620)

    

