# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualizations from natural language requests.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from datetime import datetime
import io
import shutil
import hashlib
import asyncio

import streamlit as st
import pandas as pd
import plotly.io as pio
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# Create PDF storage directory
PDF_DIR = Path("./uploaded_pdfs")
PDF_DIR.mkdir(exist_ok=True)

def save_uploaded_pdfs(uploaded_files):
    """Save uploaded PDF files to the PDF directory"""
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = PDF_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
        logger.info(f"PDF saved: {file_path}")
    return saved_files

def create_pdf_vector_store():
    """Create vector store from all PDF files in the PDF directory"""
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        return None
    
    vector_store = client.vector_stores.create(name="PDF Documents Store")
    
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            uploaded_file_obj = client.files.create(file=file, purpose="assistants")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=uploaded_file_obj.id
            )
    
    return vector_store.id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# * APP INPUTS ----

MODEL_LIST = ["gpt-4.1-mini-2025-04-14","gpt-4o-mini", "gpt-4o"]
TITLE = "Investment Analyst AI Agent"

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)
st.title(TITLE)

# Load environment variables from .env file
load_dotenv()

# Try to get API key from environment variable first
env_api_key = os.getenv("OPENAI_API_KEY")
if env_api_key:
    st.session_state["OPENAI_API_KEY"] = env_api_key
    #st.sidebar.success("✅ Using API key from environment variable")

# Test OpenAI API Key
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        #st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()


# * OpenAI Model Selection
st.sidebar.header("Choose OpenAI Model")
model_option = st.sidebar.selectbox("Select Model", MODEL_LIST, index=0)

# Intelligent Analysis Mode
st.sidebar.header("Analysis Mode")
use_intelligent_agent = st.sidebar.checkbox(
    "Enable Intelligent Analysis Mode",
    value=True,  # Set default value directly
    key="intelligent_mode",
    help="Enable Intelligent Analysis Mode: Agent will decide on analysis metrics and generate comprehensive reports"
)
# Web URLs for Analysis
st.sidebar.header("🌐 Web Pages for Analysis")
web_urls_text = st.sidebar.text_area(
    "Enter URLs (one per line)",
    value="https://www.google.com/finance/quote/AMZN:NASDAQ",
    placeholder="https://example.com/article1\nhttps://example.com/article2",
    height=68,
    help="Add specific web page URLs to include in analysis"
)

# Parse URLs from text area
if "analysis_urls" not in st.session_state:
    st.session_state["analysis_urls"] = []
    
if web_urls_text:
    urls = [url.strip() for url in web_urls_text.split('\n') if url.strip()]
    st.session_state["analysis_urls"] = urls


# PDF Documents Upload Section
st.sidebar.header("📄 PDF Documents Upload")
#st.sidebar.markdown("Upload PDF documents for industry context analysis")

uploaded_pdfs = st.sidebar.file_uploader(
    "Choose PDF files", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Upload multiple PDF files for comprehensive industry analysis"
)

# Initialize processed files tracking
if "processed_pdf_hashes" not in st.session_state:
    st.session_state.processed_pdf_hashes = set()

# Handle PDF uploads with duplicate prevention
if uploaded_pdfs:
    current_hashes = set()
    new_files = []
    
    for pdf in uploaded_pdfs:
        file_hash = hashlib.md5(pdf.getbuffer()).hexdigest()
        current_hashes.add(file_hash)
        if file_hash not in st.session_state.processed_pdf_hashes:
            new_files.append(pdf)
    
    if new_files:
        with st.sidebar:
            with st.spinner("Processing PDF files..."):
                saved_files = save_uploaded_pdfs(new_files)
                if saved_files:
                    st.success(f"✅ {len(saved_files)} new PDF file(s) uploaded!")
                    vector_store_id = create_pdf_vector_store()
                    if vector_store_id:
                        st.session_state["pdf_vector_store_id"] = vector_store_id
                        st.success("✅ PDF vector store updated!")
        
        # Update processed hashes
        st.session_state.processed_pdf_hashes.update(current_hashes)

# Display current PDF files
current_pdfs = list(PDF_DIR.glob("*.pdf"))
if current_pdfs:
    st.sidebar.markdown("**Current PDF Files:**")
    for pdf_file in current_pdfs:
        st.sidebar.text(f"📄 {pdf_file.name}")
    
    if st.sidebar.button("🗑️ Clear All PDFs"):
        for pdf_file in current_pdfs:
            pdf_file.unlink()
        st.session_state.pop("pdf_vector_store_id", None)
        st.session_state.processed_pdf_hashes.clear()
        st.sidebar.success("All PDF files cleared!")
        st.rerun()

# ---------------------------
# File Upload and Data Preview
# ---------------------------

# * Analysis Request Configuration
st.sidebar.header("Analysis Request")
analysis_request = st.sidebar.text_area(
    "Customize Analysis Requirements",
    value="""1 .Executive Summary & Recommendation (Minimum 100 words): A high-level summary of the investment thesis and the agent's recommendation. This section briefly states the company's name, ticker, current price, and the recommended action. It includes the target price and a timeframe. It also summarizes key reasons for the recommendation. 
2. Company Overview (Minimum 100 words): Background on the company's business model, main products/services, and revenue segments. This gives context about what the company does and its position in the market. It may mention the company's market capitalization, sector, and any recent significant developments. Industry & Market Trends: Discussion of the industry in which the company operates, including current trends, competitive landscape, and macroeconomic factors. 
3. Fundamental Analysis (Financial Performance) (Minimum 200 words): An evaluation of the company's financial health and performance using fundamental data. Key metrics: Growth,Profitability, Returns, Valuation Multiples, Financial Health.
4. Technical Analysis (Minimum 200 words): An examination of the stock's recent price action and chart patterns. The agent should include key technical indicators and trends:
Price Trend, Moving Averages, Momentum Indicators.
5. Sentiment Analysis (Minimum 100 words): Evaluation of market sentiment around the stock. This includes analyzing news sentiment, social media chatter.
6. Risk Factors & Considerations (Minimum 200 words): A frank discussion of the risks involved in the investment. The agent lists key risk factors such as: increasing competition, regulatory risks, macroeconomic factors, company-specific issues, or execution risks. 
7. Conclusion (Minimum 100 words): A wrap-up that reiterates the investment recommendation and key points. It should tie together the analysis from previous sections. 
    """,
    height=600,
    help="Customize the analysis requirements covered in the report."
)

# Cache LLM initialization
if "llm" not in st.session_state or st.session_state.get("cached_model") != model_option:
    st.session_state["llm"] = ChatOpenAI(model=model_option, api_key=st.session_state["OPENAI_API_KEY"])
    st.session_state["cached_model"] = model_option
llm = st.session_state["llm"]

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Initialize PDF vector store if not already done and PDFs exist
    if "pdf_vector_store_id" not in st.session_state:
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if pdf_files:
            with st.spinner("Setting up PDF analysis environment..."):
                vector_store_id = create_pdf_vector_store()
                if vector_store_id:
                    st.session_state["pdf_vector_store_id"] = vector_store_id

# If no CSV file is uploaded, show message and stop
if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to get started.")
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

def display_chat_history():
    final_report = None
    analysis_results = []
    question = None
    
    # 首先遍历消息收集数据
    for msg in msgs.messages:
        if msg.type == "human":
            question = msg.content
        elif "📑 **Final Analysis Report**:" in msg.content:
            # 提取最终报告内容
            final_report = msg.content.replace("📑 **Final Analysis Report**: ", "")
    
    # 然后正常显示消息
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                plot_data = st.session_state.plots[plot_index]
                
                # Convert to Plotly object only when displaying
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                
                st.plotly_chart(plot_obj, key=f"history_plot_{plot_index}")
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                )
            elif "📑 **Final Analysis Report**:" in msg.content:
                # 显示最终报告并在同一消息中添加下载按钮
                st.write(msg.content)
                
                # 在同一消息中添加PDF下载按钮
                if final_report and question:
                    # 从session state获取analysis_results
                    pdf_data = st.session_state.get('pdf_report_data', {})
                    analysis_results = pdf_data.get('analysis_results', None)
                    
                    pdf_buffer = generate_pdf_report(
                        None, 
                        final_report,
                        question,
                        "intelligent_analysis",
                        analysis_results  # 使用完整的analysis_results包含图表
                    )
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=pdf_buffer,
                        file_name=f"Investment_Intelligent_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key=f"persistent_download_btn_{idx}"  # 使用消息索引作为key
                    )
            else:
                st.write(msg.content)

# ---------------------------
# Enhanced Analysis Functions
# ---------------------------

def generate_pdf_report(data, analysis_text, question, report_type, analysis_results=None):
    """Generate PDF report with data and analysis"""
    import plotly.io as pio
    import plotly.graph_objects as go
    from reportlab.platypus import Image, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    import tempfile
    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import requests
    import json
    import io
    
    # 简化的中文字体注册
    font_name = 'NotoSansCJK'
    if font_name not in pdfmetrics.getRegisteredFontNames():
        try:
            # 使用TTF格式的中文字体
            font_url = "https://github.com/life888888/cjk-fonts-ttf/releases/download/v0.1.0/NotoSansCJKsc-Regular.ttf"
            response = requests.get(font_url, timeout=30)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as f:
                f.write(response.content)
                pdfmetrics.registerFont(TTFont(font_name, f.name))
                os.unlink(f.name)
        except:
            # 回退到默认字体
            font_name = 'Helvetica'
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 设置常用样式使用中文字体
    if font_name != 'Helvetica':
        styles['Normal'].fontName = font_name
        styles['Heading1'].fontName = font_name
        styles['Heading2'].fontName = font_name
        styles['Heading3'].fontName = font_name
    
    # Create custom style with 1.5 line spacing
    custom_normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        leading=styles['Normal'].fontSize * 1.5,
        spaceAfter=6
    )
    
    story = []
    temp_files = []  # Track temporary files for cleanup
    
    def process_markdown_text(text, normal_style=custom_normal_style):
        """Convert basic markdown to reportlab paragraphs"""
        paragraphs = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                paragraphs.append(Spacer(1, 6))
                continue
                
            # Handle headers
            if line.startswith('### '):
                content = line[4:].strip()
                paragraphs.append(Paragraph(content, styles['Heading3']))
            elif line.startswith('## '):
                content = line[3:].strip()
                paragraphs.append(Paragraph(content, styles['Heading2']))
            elif line.startswith('# '):
                content = line[2:].strip()
                paragraphs.append(Paragraph(content, styles['Heading1']))
            # Handle bold text
            elif '**' in line:
                import re
                content = line
                # Handle paired **text** 
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                # Handle single **Title: at start
                content = re.sub(r'\*\*([^*]+?):', r'<b>\1:</b>', content)
                paragraphs.append(Paragraph(content, normal_style))
            # Handle bullet points
            elif line.startswith('- ') or line.startswith('• '):
                import re
                content = line[2:].strip()
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                content = re.sub(r'\*\*([^*]+?):', r'<b>\1:</b>', content)
                paragraphs.append(Paragraph(f"• {content}", normal_style))
            # Handle numbered lists
            elif line and line[0].isdigit() and '. ' in line:
                import re
                content = line.split('. ', 1)[1] if '. ' in line else line
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                content = re.sub(r'\*\*([^*]+?):', r'<b>\1:</b>', content)
                paragraphs.append(Paragraph(content, normal_style))
            # Regular text
            else:
                import re
                content = line
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                content = re.sub(r'\*\*([^*]+?):', r'<b>\1:</b>', content)
                paragraphs.append(Paragraph(content, normal_style))
                
        return paragraphs
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle', 
        parent=styles['Heading1'], 
        fontName=font_name,
        spaceAfter=20
    )
    story.append(Paragraph(f"Investment Data Analysis Report - {report_type.title()}", title_style))
    story.append(Spacer(1, 12))
    
    # Question
    story.append(Paragraph(f"<b>Question:</b> {question}", custom_normal_style))
    story.append(Spacer(1, 12))
    
    # Add charts and tables from analysis results (for intelligent analysis mode)
    if analysis_results:
        for result in analysis_results:
            if result['type'] == 'chart':
                # Convert dictionary to JSON string if needed
                if isinstance(result['data'], dict):
                    plot_json = json.dumps(result['data'])
                else:
                    plot_json = result['data']
                
                # Create Plotly figure from JSON and convert to image
                plot_obj = pio.from_json(plot_json)
                tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                temp_files.append(tmp_file.name)
                
                img_bytes = pio.to_image(plot_obj, format='png', width=600, height=400)
                tmp_file.write(img_bytes)
                tmp_file.flush()
                tmp_file.close()
                story.append(Image(tmp_file.name, width=400, height=300))
                story.append(Spacer(1, 12))
                    
            elif result['type'] == 'table':
                # Add table title and description (same as page display)
                story.append(Paragraph(f"<b>{result['metric']}</b>: {result['description']}", custom_normal_style))
                story.append(Spacer(1, 6))
                
                data_df = result['data']
                if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                    table_data = [data_df.columns.tolist()]
                    for _, row in data_df.head(20).iterrows():
                        table_data.append([str(val) for val in row.tolist()])
                    
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, -1), font_name),
                        ('FONTSIZE', (0, 0), (-1, 0), 8),
                        ('FONTSIZE', (0, 1), (-1, -1), 7),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 20))
    
    # Data Table (for original analysis mode)
    elif isinstance(data, pd.DataFrame) and not data.empty:
        story.append(Paragraph("<b>Data Analysis:</b>", styles['Heading2']))
        
        # Convert DataFrame to table data
        table_data = [data.columns.tolist()]
        for _, row in data.head(20).iterrows():  # Limit to first 20 rows
            table_data.append([str(val) for val in row.tolist()])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
    
    # Analysis with markdown processing
    story.append(Paragraph("<b>Comprehensive Analysis Report:</b>", styles['Heading2']))
    story.extend(process_markdown_text(analysis_text))
    
    try:
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    finally:
        # Clean up temporary files after PDF is built
        for temp_file_path in temp_files:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as cleanup_error:
                # Log cleanup error but don't fail the PDF generation
                logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")

def query_pdf_content(user_question, data_summary, model_option, analysis_request):
    """Query PDF content using the created vector store"""
    vector_store_id = st.session_state.get("pdf_vector_store_id")
    if not vector_store_id:
        return "PDF analysis not available - please upload PDF documents first"
    
    try:
        client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
        response = client.responses.create(
            model=model_option,
            input=f"""Based on the USER QUESTION, DATA SUMMARY, and SPECIFIC ANALYSIS REQUIREMENTS, search the uploaded PDF documents for relevant sections about:
    
    **USER QUESTION**: {user_question}
    **DATA PATTERNS**: {data_summary}
    **SPECIFIC ANALYSIS REQUIREMENTS**:{analysis_request}
    

    Return only relevant excerpts with clear context.""",
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id]
            }]
        )
        
        # Clean the PDF content before returning
        raw_content = response.output[1].content[0].text
        return raw_content
    except Exception as e:
        logger.error(f"PDF query error: {e}")
        return f"PDF analysis temporarily unavailable: {str(e)}"

async def scrape_specific_urls(urls, user_question="", analysis_request=""):
    """Scrape content from specific URLs using LangChain with intelligent extraction"""
    if not urls:
        return ""
    
    try:
        # Load HTML content from URLs
        loader = AsyncHtmlLoader(urls)
        docs = await loader.aload()
        
        # Transform HTML to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        
        # Combine and intelligently extract content
        scraped_content = []
        for i, doc in enumerate(docs_transformed):
            # Use intelligent extraction if user question is provided
            if user_question:
                extracted_content = extract_relevant_content(
                    doc.page_content, user_question, analysis_request
                )
            else:
                extracted_content = doc.page_content[:1500]  # Fallback to simple truncation
            
            scraped_content.append(f"\n**Content from {urls[i]}:**\n{extracted_content}")
        
        return "\n".join(scraped_content)
    except Exception as e:
        logger.error(f"Error scraping URLs: {e}")
        return ""

def extract_relevant_content(content, user_question, analysis_request=""):
    """Extract relevant content from webpage using LLM-based intelligent extraction"""
    # Split content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(content)
    
    # Use LLM to extract relevant information based on user question
    extraction_prompt = f"""
    From the following web content, extract only the information that is relevant to user's question and analysis requirements:
    
    User Question: {user_question}
    Analysis requirements: {analysis_request}...
    Web Content:
    {chunks[0]}
    
    Instructions:
    - Extract only sentences/paragraphs directly related to the question
    - Focus on financial data, market analysis, company information, or investment insights
    - Ignore navigation menus, ads, footer content, and unrelated information  
    - Summarize in 2-3 concise paragraphs (max 400 words)
    - If no relevant information found, return "No relevant financial information found"
    
    Relevant Information:
    """
    
    try:
        # Get LLM from session state
        llm = st.session_state.get("llm")
        if llm:
            response = llm.invoke(extraction_prompt)
            extracted = response.content.strip()
            logger.info(f"Extracted content: {extracted}")
            # Return extracted content if meaningful, otherwise truncate original
            if len(extracted) > 50 and "No relevant" not in extracted:
                return extracted
        
        # Fallback: return first chunk
        return chunks[0]
        
    except Exception as e:
        logger.error(f"Error in intelligent extraction: {e}")
        return chunks[0]

def search_web_context(query, data_summary, model_option, analysis_request):
    """Search web for relevant payment industry context using OpenAI's built-in web search"""
    combined_results = []
    
    # 1. Regular web search
    try:
        # Create ChatOpenAI instance with web search tool
        search_llm = ChatOpenAI(
            model=model_option,  # Use a model that supports web search
            api_key=st.session_state["OPENAI_API_KEY"]
        )
        
        # Bind web search tool to the model
        search_tool = {"type": "web_search_preview"}
        llm_with_search = search_llm.bind_tools([search_tool])
        
        # Create search query with context
        search_query = f"""Search for the reputable information in 2024 and 2025 on investment analysis related to: 
        
        **USER QUERY**: {query}
        **DATA SUMMARY**: {data_summary}
        **SPECIFIC ANALYSIS REQUIREMENTS**:{analysis_request}
        
        Fetch related key information based on the Query, analysis requirements and the data summary."""
        
        # Perform web search
        response = llm_with_search.invoke([
            {"role": "user", "content": f"Search for recent information about: {search_query}"}
        ])
        
        # Extract search results from response
        if hasattr(response, 'content') and response.content:
            # Handle content blocks if they exist
            if isinstance(response.content, list):
                search_results = []
                for block in response.content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        search_results.append(block.get('text', ''))
                web_search_content = "\n".join(search_results) if search_results else f"Search performed for: {query}"
            else:
                web_search_content = response.content
        else:
            web_search_content = f"Search performed for: {query}"
            
        if web_search_content:
            combined_results.append("**General Web Search Results:**\n" + web_search_content)
            
    except Exception as e:
        # Fallback if web search fails
        print(f"Web search error: {e}")
        combined_results.append(f"Web search context: Payment industry analysis for {query}")
    
    # 2. Scrape specific URLs if provided
    if st.session_state.get("analysis_urls"):
        import asyncio
        try:
            # Get event loop or create new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async scraping with intelligent extraction
            url_content = loop.run_until_complete(
                scrape_specific_urls(
                    st.session_state["analysis_urls"], 
                    query, 
                    analysis_request
                )
            )
            
            if url_content:
                combined_results.append("\n**Content from Specified URLs:**" + url_content)
        except Exception as e:
            logger.error(f"Error scraping specific URLs: {e}")
    
    # Combine all results
    return "\n\n".join(combined_results) if combined_results else "No web content available"

def generate_enhanced_analysis(result_data, user_question, routing_type, analysis_request):
    """Generate enhanced analysis based on data results and external sources"""
    
    # First, extract data insights to prepare data summary
    if routing_type == "chart":
        try:
            data_summary = f"Chart visualization data: {str(result_data)}"
        except:
            data_summary = "Chart visualization showing data trends and patterns"
    else:
        try:
            if hasattr(result_data, 'to_string'):
                data_summary = f"Data analysis results:\n{result_data.to_string()}"
            else:
                data_summary = f"Data table: {str(result_data)}"
        except:
            data_summary = f"Data table with {len(result_data) if hasattr(result_data, '__len__') else 'multiple'} records"
    
    # Query PDF content intelligently based on question and data summary
    pdf_content = query_pdf_content(user_question, data_summary, model_option, analysis_request)
    
    # Get web search context
    web_context = search_web_context(user_question, data_summary, model_option, analysis_request)
    
    # Create comprehensive analysis prompt
    analysis_prompt = f"""
    You are an expert investment data analyst specialized in stock and risk analysis. Analyze the provided dataset and generate a brief yet insight-rich analytical summary in Chinese that effectively uncovers the value embedded in the data based on the user's question. 200 words are enough.
    
    The data is about the daily stock's open, high, low, close (OHLC) prices and trading volume in last one year from on google finance for a stock, the stock name is defined in USER QUESTION.
    
    **USER QUESTION**: {user_question}
    
    **DATA ANALYSIS RESULTS**:
    {data_summary}
    
    **PDF DOCUMENT CONTENT** (Industry Context):
    {pdf_content}
    
    **WEB SEARCH CONTEXT**:
    {web_context}
    Make this analysis practical, accurate and business-focused. Use specific data points and industry knowledge.
    """
    
    try:
        analysis_response = llm.invoke(analysis_prompt)
        return analysis_response.content
    except Exception as e:
        return f"Enhanced analysis temporarily unavailable: {str(e)}"

# ---------------------------
# Tool Input Models
# ---------------------------

class AnalyzeMetricInput(BaseModel):
    metric: str = Field(description="Descriptive name of the metric analysis to perform (e.g., 'Monthly Sales Amount Trends', 'Customer Count Growth Analysis')")
    analysis_type: str = Field(description="Type of analysis: 'table' or 'chart'")
    instructions: str = Field(description="Specific analysis requirements")

class SearchPdfInput(BaseModel):
    query: str = Field(description="Search question for PDF documents")
    data_context: str = Field(description="Data context for the search")

class SearchWebInput(BaseModel):
    query: str = Field(description="Search question for web search")
    data_context: str = Field(description="Data context for the search")

class SaveResultInput(BaseModel):
    type: str = Field(description="Type of result: 'chart', 'table', or 'text'")
    data: str = Field(description="Data content to save")
    description: str = Field(description="Description of the result")

# ---------------------------
# Intelligent Data Analysis Agent
# ---------------------------

class IntelligentDataAnalysisAgent:
    """Intelligent Data Analysis Agent that can autonomously decide analysis metrics and generate reports"""
    
    def __init__(self, llm, data_analyst, analysis_request):
        self.llm = llm
        self.data_analyst = data_analyst
        self.analysis_request = analysis_request
        self.analysis_results = []
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
    def _create_tools(self):
        """Create tools used by the Agent"""
        tools = [
            StructuredTool.from_function(
                name="analyze_metric",
                description="Use descriptive metric names that explain the analysis (e.g., 'Monthly Sales Trends', 'Year-over-Year Growth Analysis')",
                func=self._analyze_metric,
                args_schema=AnalyzeMetricInput
            ),
            StructuredTool.from_function(
                name="search_pdf",
                description="Search for relevant information in PDF documents",
                func=self._search_pdf,
                args_schema=SearchPdfInput
            ),
            StructuredTool.from_function(
                name="search_web",
                description="Search for relevant information on the web",
                func=self._search_web,
                args_schema=SearchWebInput
            ),
            StructuredTool.from_function(
                name="save_analysis_result",
                description="Save analysis results",
                func=self._save_result,
                args_schema=SaveResultInput
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create Agent executor"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a professional financial analyst assistant specialized in stock investment and risk analysis. Your analysis strategy depends on the user's input. The user's input will be sent to you in the analyze function.

ANALYSIS STRATEGY:
1. **If user asks specific metrics** (e.g., "Monthly Average Price", "Relative Strength Index"):
   - Focus primarily on the user-requested metrics
   - Use analyze_metric tool for the requested metrics first
   - If token budget allows, add additional valuable related metrics to a total of 3-5 metrics
   
2. **If user asks general questions** without specific metrics:
    - Call the analyze Identify 3-5 most important EXISTING data columns to analyze.
    - Create descriptive metric names that clearly explain what analysis will be performed (e.g., if analyzing "Moving Average" column by month, use "Monthly Moving Average Trends" as metric name; if doing Volume by Price Buckets, use "Trade Volume by Price Level")
    - Metric names should describe the specific analysis type (trends, comparisons, distributions, etc.)
    - ONLY analyze actual columns that exist in the dataset, but name the metrics descriptively
    - Use the analyze_metric tool to generate relevant charts and tables
    - Finally generate a comprehensive report containing multiple views and in-depth analysis


Analysis report requirements:
{self.analysis_request}

- All analysis must be based on real, existing data. User will provide the data summary in the analysis request.
- Do not analyze the same metric or similar metrics twice.

Important notes:
- Consider whether to use charts or tables for each metric, prefer charts but you must have at least 1 table. 
- For the table metrics, focus on summary statistics, top/bottom performers, or aggregated data rather than raw detailed records.
- Ensure analysis focuses on the company mentioned in the user's question.

METRIC NAMING GUIDELINES:
- Use descriptive, business-friendly metric names instead of just column names
- Include the analysis type in the metric name (e.g., "Trends", "Comparison", "Distribution", "Growth Analysis")
- Examples of good metric names:
  * "Monthly Average Price" (instead of just "Average Price")
  * "Monthly Volatility" (instead of just "Volatility") 
  * "Daily Price Range" (instead of just "Price Range")
  * "Number of Up Days vs Down Days" (instead of just "Days vs Down Days")

FAILURE HANDLING:
- If analyze_metric returns a message starting with "FAILED:", that metric analysis failed
- When a metric fails, immediately try a different metric with different columns/analysis approach
- Continue until you have successfully analyzed enough metrics
- Do not retry the same failed metric
"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True
        )
    
    def _analyze_metric(self, metric: str, analysis_type: str, instructions: str) -> str:
        """
        Analyze specific metrics using dual instruction strategy.
        
        This method uses two types of instructions:
        1. Simple technical instructions - sent to underlying data analyst to prevent synthetic data generation
        2. Original business instructions - preserved for UI display and final reports
        """
        try:
            # Generate simple, clean technical instructions for underlying data analyst
            # This prevents the agent from creating fake data based on complex business semantics
            if analysis_type.lower() == "table":
                simple_instructions = f"Generate a summary table for {metric} analysis" 
            else:  # chart
                simple_instructions = f"Generate a chart for {metric} analysis"
            
            # Use simple technical instructions to invoke data analyst
            # This ensures the agent only works with existing data columns
            self.data_analyst.invoke_agent(
                user_instructions=simple_instructions,
                data_raw=st.session_state.get('current_df', pd.DataFrame())
            )
            result = self.data_analyst.get_response()
            logger.info(f"instructions: {instructions}")
            logger.info(f"metric: {metric}")
            # Save results for subsequent use
            routing = result.get("routing_preprocessor_decision")
            if routing == "chart":
                plot_data = result.get("plotly_graph")
                if plot_data and plot_data not in [None, {}, "null", ""]:
                    # Check if plot_data has actual content
                    try:
                        if isinstance(plot_data, dict) and plot_data.get("data"):
                            self.analysis_results.append({
                                'type': 'chart',
                                'metric': metric,
                                'data': plot_data,
                                'description': instructions
                            })
                            return f"Successfully generated chart analysis for {metric}"
                    except:
                        pass
                return f"FAILED: Empty or invalid chart generated for {metric}. Try a different metric."
            else:
                data = result.get("data_wrangled")
                if data is not None:
                    # Ensure data is in DataFrame format for PDF generation
                    if not isinstance(data, pd.DataFrame):
                        data = pd.DataFrame(data)
                    # Check if DataFrame is not empty
                    if not data.empty and len(data) > 0:
                        self.analysis_results.append({
                            'type': 'table', 
                            'metric': metric,
                            'data': data,
                            'description': instructions
                        })
                        return f"Successfully generated table analysis for {metric}"
                return f"FAILED: Empty or invalid table generated for {metric}. Try a different metric."
            
        except Exception as e:
            return f"FAILED: Error analyzing {metric}: {str(e)}. Try a different metric."
    
    def _search_pdf(self, query: str, data_context: str) -> str:
        """Search PDF content"""
        try:
            result = query_pdf_content(query, data_context, model_option, self.analysis_request)
            return f"PDF search results: {result}"
        except Exception as e:
            return f"PDF search error: {str(e)}"
    
    def _search_web(self, query: str, data_context: str) -> str:
        """Search web content"""
        try:
            result = search_web_context(query, data_context, model_option, self.analysis_request)
            return f"Web search results: {result}"
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    def _save_result(self, type: str, data: str, description: str) -> str:
        """Save analysis results"""
        try:
            # Results are already saved in _analyze_metric
            return "Results saved successfully"
        except Exception as e:
            return f"Error saving results: {str(e)}"
    
    def analyze(self, user_question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute intelligent analysis"""
        # Save current dataframe for tool usage
        st.session_state['current_df'] = df
        
        # Get data overview
        data_info = f"""
Dataset contains {len(df)} rows and {len(df.columns)} columns.
Column names: {', '.join(df.columns.tolist())}
Data types: {df.dtypes.to_dict()}
Numerical column statistics: {df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else 'No numerical columns'}
"""
        logger.info(f"data_info: {data_info}")
        logger.info(f"user_question: {user_question}")
        # Build analysis request
        analysis_prompt = f"""
        User question: {user_question}
        The data is already uploaded by user. The data overview is: {data_info}
Based on the following dataset information, please analyze key stock investiment and risk analysis metrics for the company mentioned in the user's question:

Follow the ANALYSIS STRATEGY in your system prompt:
- If the user specified particular metrics in their question, focus on those first
- If the user asked a general question, such as "Analyze a specific stock", automatically identify the most relevant business metrics based on the data summary
- Use analyze_metric tool for each metric (decide whether to use charts or tables)

Analyze the related data based on the user's report's requirements:
{analysis_request}
"""
        
        # Execute Agent analysis
        try:
            response = self.agent_executor.invoke({
                "input": analysis_prompt,
                "chat_history": []
            })
            
            # Generate final report
            final_report = self._generate_final_report(user_question)
            logger.info(f"final_report: {final_report}")
            return {
                "success": True,
                "analysis_results": self.analysis_results,
                "final_report": final_report,
                "agent_output": response.get("output", "")
            }
            
        except Exception as e:
            logger.error(f"Agent analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_results": self.analysis_results
            }
    
    def _generate_final_report(self, user_question: str) -> str:
        """Generate final comprehensive report"""
        # Collect summary and data of all analysis results
        results_summary = []
        data_summary = ""
        for result in self.analysis_results:
            if result['type'] == 'table':
                data = result['data']
                if isinstance(data, pd.DataFrame):
                    summary = f"- **{result['metric']}**: {result['description']}\n  Data:\n{data.to_string()}"
                    data_summary += f"{result['metric']}: {data.to_string()}... "
                else:
                    summary = f"- **{result['metric']}**: {result['description']}\n  Data: {str(data)}"
                    data_summary += f"{result['metric']}: {str(data)}... "
            else:
                summary = f"- **{result['metric']}**: {result['description']} (chart)\n  Chart data: {str(result['data'])}"
                data_summary += f"{result['metric']}: chart visualization data. "
            results_summary.append(summary)
        #logger.info(f"data_summary: {data_summary}")
        # Query PDF content and web search for intelligent analysis
        pdf_content = query_pdf_content(user_question, data_summary, model_option, self.analysis_request)
        web_context = search_web_context(user_question, data_summary, model_option, self.analysis_request)
        
        # Build report generation prompt
        report_prompt = f"""
        You are an expert stock investment and risk analyst. Analyze the provided dataset and generate a comprehensive and detailed professional investment and risk analysis report in Chinese. 

User question: {user_question}

    **PDF DOCUMENT CONTENT** (Complete Industry Context):
    {pdf_content}
    
    **WEB SEARCH CONTEXT**:
    {web_context}

Completed analysis with data:
{chr(10).join(results_summary)}

Analysis requirements:
{self.analysis_request}

    **FORMATTING REQUIREMENTS**:
        - Use clear markdown formatting for better readability
        - Structure your response with headers: # for main sections, ## for subsections, ### for sub-subsections
        - Use **bold text** for key findings and important metrics
        - Use bullet points (-) for lists and key points
        - Use numbered lists (1., 2., 3.) for sequential recommendations or steps
        - Ensure proper paragraph breaks for readability
        - **IMPORTANT**: To prevent markdown/LaTeX rendering issues, always escape these special characters:
          * Dollar signs: write \\$ instead of $ (e.g., "\\$6 million" not "$6 million")
          * Percent signs: write \\% instead of % (e.g., "15\\% growth" not "15% growth")
          * Ampersands: write \\& instead of & (e.g., "Johnson \\& Johnson" not "Johnson & Johnson")
          * Underscores: write \\_ instead of _ (e.g., "Sales\\_Amount" not "Sales_Amount")
          * Carets: write \\^ instead of ^ (e.g., "Q3\\^2024" not "Q3^2024")
          * Hash symbols: write \\# instead of # (e.g., "\\#1 client" not "#1 client")
        - Example structure:
        # Executive Summary
        ## Key Findings
        ### Performance Metrics
        - **Revenue Growth**: 15% YoY increase
        - **Market Share**: Expanded by 2.3%
        
        Make this analysis practical and business-focused. Use specific data points and industry knowledge.
        Don't need to add Prepared by: or other similar texts at the end of the report.
"""
        
        try:
            response = self.llm.invoke(report_prompt)
            return response.content
        except Exception as e:
            return f"Error generating report: {str(e)}"

# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = False

# Cache PandasDataAnalyst initialization - recreate if model changed
if "pandas_data_analyst" not in st.session_state or st.session_state.get("cached_model") != model_option:
    st.session_state["pandas_data_analyst"] = PandasDataAnalyst(
        model=llm,
        data_wrangling_agent=DataWranglingAgent(
            model=llm,
            log=LOG,
            bypass_recommended_steps=True,
            n_samples=100,
        ),
        data_visualization_agent=DataVisualizationAgent(
            model=llm,
            n_samples=100,
            log=LOG,
        ),
    )
pandas_data_analyst = st.session_state["pandas_data_analyst"]

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# Handle new user input
if question := st.chat_input("Enter your question here:", key="query_input"):
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()

    msgs.add_user_message(question)
    
    with st.chat_message("human"):
        st.write(question)
    
    with st.chat_message("ai"):
        # Use intelligent analysis mode
        if use_intelligent_agent:
            with st.spinner("Conducting comprehensive analysis... This may take a few minutes."):
                try:
                    # Create intelligent analysis Agent
                    intelligent_agent = IntelligentDataAnalysisAgent(
                        llm=llm,
                        data_analyst=pandas_data_analyst,
                        analysis_request=analysis_request
                    )
                    
                    # Execute intelligent analysis
                    analysis_result = intelligent_agent.analyze(question, df)
                    
                    if analysis_result["success"]:
                        # 保存PDF数据到session state以便持久显示下载按钮
                        st.session_state['pdf_report_data'] = {
                            'final_report': analysis_result["final_report"],
                            'analysis_results': analysis_result["analysis_results"],
                            'question': question,
                            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                        }
                        
                        # Add completion message
                        msgs.add_ai_message("📊 **Comprehensive Analysis Complete**")
                        
                        # Process each analysis result
                        for idx, result in enumerate(analysis_result["analysis_results"]):
                            if result['type'] == 'chart':
                                msgs.add_ai_message(f"**{result['metric']}**: {result['description']}")
                                
                                # Store chart data directly without immediate conversion
                                plot_index = len(st.session_state.plots)
                                st.session_state.plots.append(result['data'])
                                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                                
                            elif result['type'] == 'table':
                                msgs.add_ai_message(f"**{result['metric']}**: {result['description']}")
                                
                                # Process table data
                                data = result['data']
                                if not isinstance(data, pd.DataFrame):
                                    data = pd.DataFrame(data)
                                    
                                # Store table
                                df_index = len(st.session_state.dataframes)
                                st.session_state.dataframes.append(data)
                                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                        
                        # Add final report
                        msgs.add_ai_message(f"📑 **Final Analysis Report**: {analysis_result['final_report']}")
                        
                    else:
                        error_msg = f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"
                        msgs.add_ai_message(error_msg)
                        
                except Exception as e:
                    error_msg = f"Intelligent analysis error: {str(e)}"
                    msgs.add_ai_message(error_msg)
                    logger.error(f"Intelligent Agent Error: {e}")
                    
        # Use original analysis mode        
        else:
            with st.spinner("Conducting general analysis..."):
                try:
                    pandas_data_analyst.invoke_agent(
                        user_instructions=question,
                        data_raw=df,
                    )
                    result = pandas_data_analyst.get_response()
                except Exception as e:
                    error_msg = f"An error occurred while processing your query: {str(e)}"
                    msgs.add_ai_message(error_msg)
                    print(f"Pandas Data Analyst Error: {e}") 
                    st.stop()

                routing = result.get("routing_preprocessor_decision")

                if routing == "chart" and not result.get("plotly_error", False):
                    # Process chart result
                    plot_data = result.get("plotly_graph")
                    if plot_data:
                        # Convert dictionary to JSON string if needed
                        if isinstance(plot_data, dict):
                            plot_json = json.dumps(plot_data)
                        else:
                            plot_json = plot_data
                        plot_obj = pio.from_json(plot_json)
                        response_text = "Returning the generated chart."
                        # Store the chart
                        plot_index = len(st.session_state.plots)
                        st.session_state.plots.append(plot_obj)
                        msgs.add_ai_message(response_text)
                        msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                        
                        # Enhanced Analysis
                        analysis = generate_enhanced_analysis(plot_data, question, "chart", analysis_request)
                        msgs.add_ai_message(f"📊 **Enhanced Analysis:** {analysis}")
                            
                    else:
                        msgs.add_ai_message("The agent did not return a valid chart.")

                elif routing == "table":
                    # Process table result
                    data_wrangled = result.get("data_wrangled")
                    if data_wrangled is not None:
                        response_text = "Returning the data table."
                        # Ensure data_wrangled is a DataFrame
                        if not isinstance(data_wrangled, pd.DataFrame):
                            data_wrangled = pd.DataFrame(data_wrangled)
                        df_index = len(st.session_state.dataframes)
                        st.session_state.dataframes.append(data_wrangled)
                        msgs.add_ai_message(response_text)
                        msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                        
                        # Enhanced Analysis
                        analysis = generate_enhanced_analysis(data_wrangled, question, "table", analysis_request)
                        msgs.add_ai_message(f"📋 **Enhanced Analysis:** {analysis}")
                            
                    else:
                        msgs.add_ai_message("No table data was returned by the agent.")
                else:
                    # Fallback if routing decision is unclear or if chart error occurred
                    data_wrangled = result.get("data_wrangled")
                    if data_wrangled is not None:
                        response_text = (
                            "I apologize. There was an issue with generating the chart. "
                            "Returning the data table instead."
                        )
                        if not isinstance(data_wrangled, pd.DataFrame):
                            data_wrangled = pd.DataFrame(data_wrangled)
                        df_index = len(st.session_state.dataframes)
                        st.session_state.dataframes.append(data_wrangled)
                        msgs.add_ai_message(response_text)
                        msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                    else:
                        response_text = (
                            "An error occurred while processing your query. Please try again."
                        )
                        msgs.add_ai_message(response_text)
                        
    # Rerun to display the new messages
    st.rerun()
