# AI Data Science Team

**An AI-powered data science team that uses agents to perform common data science** tasks including data cleaning, preparation, feature engineering, modeling (machine learning), interpretation on various business problems like:

- Churn Modeling
- Employee Attrition
- Lead Scoring
- Insurance Risk
- Credit Card Risk
- And more

## Companies That Want An AI Data Science Team

If you are interested in having your own AI Data Science Team built and deployed for your enterprise, send inquiries here: [https://www.business-science.io/contact.html](https://www.business-science.io/contact.html)

## Free Generative AI For Data Scientists Workshop

If you want to learn how to build AI Agents for your company that perform Data Science, Business Intelligence, Churn Modeling, Time Series Forecasting, and more, [register for my next Generative AI for Data Scientists workshop here.](https://learn.business-science.io/ai-register)

## Agents

This project is a work in progress. New agents will be released soon.

![Data Science Team](/img/ai_data_science_team.jpg)

### Agents Available Now:

1. **Data Cleaning Agent:** Performs Data Preparation steps including handling missing values, outliers, and data type conversions.

### Agents Coming Soon:

1. **Supervisor:** Forms task list. Moderates sub-agents. Returns completed assignment. 
2. **Exploratory Data Analyst:** Analyzes data structure, creates exploratory visualizations, and performs correlation analysis to identify relationships. 
3. **Feature Engineering Agent:** Converts the prepared data into ML-ready data. Adds features to increase predictive accuracy of ML models. 
4. **Machine Learning Agent:** Builds and logs the machine learning models.
5. **Interpretability Agent:** Performs Interpretable ML to explain why the model returned predictions including which features were the most important to the model.

## Disclaimer

**This project is for educational purposes only.**

- It is not intended to replace your company's data science team
- No warranties or guarantees provided
- Creator assumes no liability for financial loss
- Consult an experienced Generative AI Data Scientist for building your own AI Data Science Team
- If you want an enterprise-grade AI Data Science Team, [send inquiries here](https://www.business-science.io/contact.html). 

By using this software, you agree to use it solely for learning purposes.

## Table of Contents

- [AI Data Science Team](#ai-data-science-team)
  - [Companies That Want An AI Data Science Team](#companies-that-want-an-ai-data-science-team)
  - [Free Generative AI For Data Scientists Workshop](#free-generative-ai-for-data-scientists-workshop)
  - [Agents](#agents)
    - [Agents Available Now:](#agents-available-now)
    - [Agents Coming Soon:](#agents-coming-soon)
  - [Disclaimer](#disclaimer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Example 1: Cleaning Data with the Data Cleaning Agent](#example-1-cleaning-data-with-the-data-cleaning-agent)
  - [Contributing](#contributing)
  - [License](#license)

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

