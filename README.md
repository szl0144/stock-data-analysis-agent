# AI Data Science Team

**An AI-powered data science team that uses agents to perform common data science** tasks including data cleaning, preparation, feature engineering, modeling (machine learning), interpretation on various business problems like:

- Churn Modeling
- Employee Attrition
- Lead Scoring
- Insurance Risk
- Credit Card Risk
- And more

## Free Generative AI Data Science Workshop

If you want to learn how to build AI Agents that perform Data Science, Business Intelligence, Churn Modeling, Time Series Forecasting, and more, [register for my next free AI for Data Scientists workshop here.](https://learn.business-science.io/ai-register)

## Agents

This project is a work in progress. Currently there is the following Agents:

1. **Data Cleaning Agent**: Performs Data Preparation steps including handling missing values, outliers, and data type conversions.


[TODO - INSERT IMAGE]

## Disclaimer

**This project is for educational purposes only.**

- It is not intended to replace your company's data science team
- No warranties or guarantees provided
- Creator assumes no liability for financial loss
- Consult an experienced Generative AI Data Scientist for building your own AI Data Science Team

By using this software, you agree to use it solely for learning purposes.

## Table of Contents

- [AI Data Science Team](#ai-data-science-team)
  - [Free Generative AI Data Science Workshop](#free-generative-ai-data-science-workshop)
  - [Agents](#agents)
  - [Disclaimer](#disclaimer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Example 1: Cleaning Data with the Data Cleaning Agent](#example-1-cleaning-data-with-the-data-cleaning-agent)
  - [Contributing](#contributing)
  - [License](#license)
  - [Free GenAI Data Science Workshop](#free-genai-data-science-workshop)

## Installation

``` bash
pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade
```

## Usage

### Example 1: Cleaning Data with the Data Cleaning Agent

[See the full example here.](https://github.com/business-science/ai-data-science-team/blob/master/examples/data_cleaning_agent.ipynb) 

``` python
data_cleaning_agent = data_cleaning_agent(model = llm, log=LOG, log_path=LOG_PATH)

response = data_cleaning_agent.invoke({
    "user_instructions": "Don't remove outliers when cleaning the data.",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})
```

``` bash
---DATA CLEANING AGENT----
    * CREATE DATA CLEANER CODE
    * EXECUTING AGENT CODE
    * EXPLAIN AGENT CODE
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details. 

## Free GenAI Data Science Workshop

If you want to learn how to build Generative AI Agents that perform Data Science, Business Intelligence, Churn Modeling, Time Series Forecasting, and more, [register for my next free AI for Data Scientists workshop here.](https://learn.business-science.io/ai-register)
