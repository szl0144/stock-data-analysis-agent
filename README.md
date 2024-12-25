# Your AI Data Science Team (An Army Of Copilots)

**An AI-powered data science team of copilots that uses agents to help you perform common data science tasks 10X faster**.

Star ⭐ This GitHub (Takes 2 seconds and means a lot).

---

The AI Data Science Team of Copilots includes Agents that specialize data cleaning, preparation, feature engineering, modeling (machine learning), and interpretation of various business problems like:

- Churn Modeling
- Employee Attrition
- Lead Scoring
- Insurance Risk
- Credit Card Risk
- And more

## Companies That Want An AI Data Science Team Copilot

If you are interested in having your own custom enteprise-grade AI Data Science Team Copilot, send inquiries here: [https://www.business-science.io/contact.html](https://www.business-science.io/contact.html)

## Free Generative AI For Data Scientists Workshop

If you want to learn how to build AI Agents for your company that performs Data Science, Business Intelligence, Churn Modeling, Time Series Forecasting, and more, [register for my next Generative AI for Data Scientists workshop here.](https://learn.business-science.io/ai-register)

## Data Science Agents

This project is a work in progress. New data science agents will be released soon.

![Data Science Team](/img/ai_data_science_team.jpg)

### Agents Available Now

1. **Data Wrangling Agent:** Merges, Joins, Preps and Wrangles data into a format that is ready for data analysis.
2. **Data Cleaning Agent:** Performs Data Preparation steps including handling missing values, outliers, and data type conversions.
3. **Feature Engineering Agent:** Converts the prepared data into ML-ready data. Adds features to increase predictive accuracy of ML models.

### Agents Coming Soon

1. **Data Analyst:** Analyzes data structure, creates exploratory visualizations, and performs correlation analysis to identify relationships.
2. **Machine Learning Agent:** Builds and logs the machine learning models.
3. **Interpretability Agent:** Performs Interpretable ML to explain why the model returned predictions including which features were the most important to the model.
4. **Supervisor:** Forms task list. Moderates sub-agents. Returns completed assignment. 

## Disclaimer

**This project is for educational purposes only.**

- It is not intended to replace your company's data science team
- No warranties or guarantees provided
- Creator assumes no liability for financial loss
- Consult an experienced Generative AI Data Scientist for building your own custom AI Data Science Team
- If you want a custom enterprise-grade AI Data Science Team, [send inquiries here](https://www.business-science.io/contact.html). 

By using this software, you agree to use it solely for learning purposes.

## Table of Contents

- [Your AI Data Science Team (An Army Of Copilots)](#your-ai-data-science-team-an-army-of-copilots)
  - [Companies That Want An AI Data Science Team Copilot](#companies-that-want-an-ai-data-science-team-copilot)
  - [Free Generative AI For Data Scientists Workshop](#free-generative-ai-for-data-scientists-workshop)
  - [Data Science Agents](#data-science-agents)
    - [Agents Available Now](#agents-available-now)
    - [Agents Coming Soon](#agents-coming-soon)
  - [Disclaimer](#disclaimer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Example 1: Feature Engineering with the Feature Engineering Agent](#example-1-feature-engineering-with-the-feature-engineering-agent)
    - [Example 2: Cleaning Data with the Data Cleaning Agent](#example-2-cleaning-data-with-the-data-cleaning-agent)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

``` bash
pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade
```

## Usage

### Example 1: Feature Engineering with the Feature Engineering Agent

[See the full example here.](/examples/feature_engineering_agent.ipynb)

``` python
feature_engineering_agent = make_feature_engineering_agent(model = llm)

response = feature_engineering_agent.invoke({
    "user_instructions": "Make sure to scale and center numeric features",
    "target_variable": "Churn",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})
```

``` bash
---FEATURE ENGINEERING AGENT----
    * CREATE FEATURE ENGINEER CODE
    * EXECUTING AGENT CODE
    * EXPLAIN AGENT CODE
```

### Example 2: Cleaning Data with the Data Cleaning Agent

[See the full example here.](/examples/data_cleaning_agent.ipynb) 

``` python
data_cleaning_agent = make_data_cleaning_agent(model = llm)

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

