# Stock Investment Analysis App

An intelligent stock investment analysis application powered by AI, featuring data visualization, PDF document analysis, and web scraping capabilities.

## Features

- 📊 **Interactive Stock Data Analysis** - Upload and analyze stock data with intelligent AI agents
- 🤖 **AI-Powered Insights** - Get comprehensive investment analysis using OpenAI models
- 📄 **PDF Document Analysis** - Upload PDF documents for contextual analysis
- 🌐 **Web Scraping Integration** - Analyze specific web pages for additional market insights
- 📋 **Detailed Reports** - Generate comprehensive PDF reports with charts and analysis
- 🎯 **Smart Routing** - Automatic analysis type detection (charts vs tables)

## Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/szl0144/stock-data-analysis-agent
cd ai-data-science-team/apps/stock-investment-analysis
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root or set environment variables:

```bash
# Option 1: Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or set the environment variable directly:

```bash
# Option 2: Set environment variable
export OPENAI_API_KEY="your_openai_api_key_here"
```

**To get your OpenAI API key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and use it in your configuration

### 5. Launch the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### Data Upload
1. **Upload Stock Data**: Use the file uploader to upload CSV files with stock data
2. **Expected Format**: CSV files should contain columns like Date, Open, High, Low, Close, Volume

### Analysis Options
- **Intelligent Analysis Mode**: Enable for AI-powered comprehensive analysis
- **Original Analysis Mode**: Use for basic data analysis and visualization

### Additional Features
- **PDF Documents**: Upload PDF files for industry context and background analysis
- **Web Pages**: Add specific URLs to include web content in your analysis
- **Custom Analysis**: Specify particular metrics or analysis requirements

### Report Generation
- View analysis results in the chat interface
- Download comprehensive PDF reports with charts and insights
- Export data visualizations

## Configuration

### Model Selection
Choose from available OpenAI models:
- GPT-4.1-mini (recommended for complex analysis)
- GPT-4o (single question analysis)

### Web Pages for Analysis
Add specific URLs in the sidebar to include external web content:
```
https://example.com/financial-report
https://finance.yahoo.com/quote/STOCK
```

## Project Structure

```
stock-investment-analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── uploaded_pdfs/        # Directory for uploaded PDF files
├── logs/                 # Application logs
└── README.md            # This file
```

## Dependencies

Key packages used:
- **streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **plotly** - Interactive data visualization
- **openai** - OpenAI API integration
- **langchain** - LLM application framework
- **reportlab** - PDF generation
- **beautifulsoup4** - Web scraping
- **aiohttp** - Async HTTP client

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: OpenAI API key not found
```
**Solution**: Ensure your API key is properly set in environment variables or .env file

**2. Package Installation Issues**
```
ERROR: Could not install packages
```
**Solution**: Try upgrading pip and reinstalling:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**3. Font Issues in PDF**
```
PDF shows black blocks instead of Chinese text
```
**Solution**: The app automatically downloads Chinese fonts. Ensure internet connection during first PDF generation.

**4. Port Already in Use**
```
Port 8501 is already in use
```
**Solution**: Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Logs
Check the `logs/` directory for detailed application logs and error messages.

## Development

### Running in Development Mode
```bash
streamlit run app.py --reload
```

### Code Structure
- `app.py` - Main application with Streamlit interface
- PDF generation with Chinese font support
- Intelligent agent system for automated analysis
- Web scraping with content extraction
- Vector store for PDF document search

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs in the `logs/` directory
3. Ensure all dependencies are properly installed
4. Verify API key configuration

## License

This project is part of the AI Data Science Team framework.

---

**Note**: This application requires an active internet connection for AI model calls, font downloads, and web scraping features. 