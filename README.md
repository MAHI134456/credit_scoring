# Credit Scoring Project
This project builds a complete credit scoring system, from data preprocessing and model training to deploying a simple prediction API using FastAPI and Docker.
It also includes a minimal CI pipeline using GitHub Actions.

## Project Structure
```bash 
├── .github/workflows/ci.yml   # CI pipeline
├── data/                      # Data folder (should be in .gitignore)
│   ├── raw/                   # Raw data files
│   └── processed/             # Cleaned & processed data
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering and cleaning
│   ├── train.py               # Model training script
│   ├── predict.py             # Inference logic
│   └── api/
│       ├── main.py            # FastAPI app
│       └── pydantic_models.py # Pydantic schemas for input/output
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile                 # Build image for app
├── docker-compose.yml         # Define services
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to ignore
└── README.md  
```

## Features
- EDA & Data Processing: Notebook and scripts for cleaning and feature engineering.
- Model Training: Train and save a machine learning model for credit scoring.
- API: Serve predictions with a FastAPI app.
- Docker: Containerize the entire app for easy deployment.
- CI/CD: Basic GitHub Actions workflow to test your code on push or pull requests.

## Quick Start

1. Clone the repo

```bash git clone https://github.com/<your-username>/credit_scoring.git  cd credit_scoring ```

2. Create a virtual environment

```bash python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate
 ```

3. Install dependencies

``` bash pip install -r requirements.txt```

4. Run tests
```bash pytest tests/```

5. Build & run with Docker
```bash docker build -t credit-scoring-app . docker run -p 8000:8000 credit-scoring-app```

## Credit Scoring Business Understanding
Credit Scoring Business Understanding  
The Basel II Accord focuses on detailed risk measurement, requiring banks to use strong methods like the Internal Ratings-Based (IRB) approach to evaluate credit risk. This need directly affects the demand for clear and well-documented credit scoring models. Clarity helps meet regulatory requirements, as regulators expect transparency in how risk assessments are made. This ensures compliance with Basel II’s Pillar 1, which deals with minimum capital needs, and Pillar 2, which involves supervisory reviews. A documented model aids in audits and confirms risk estimates, such as the probability of default (PD), to maintain capital adequacy and reduce systemic risk. Without proper documentation and clarity, banks face the risk of non-compliance, leading to fines or increased capital needs, which can stress financial resources.  
Since direct "default" labels are not available, it is necessary to create a proxy variable, such as late payments or credit utilization ratios, to estimate default risk. This proxy acts as a stand-in for real default data, allowing for model training. However, using a proxy can introduce business challenges. This includes the potential for inaccurate risk assessments if the proxies are not aligned, as late payments do not always mean a default will happen. This misalignment can result in mispriced loans, higher credit losses, or regulatory issues if the proxy does not meet Basel II’s standards for strong risk measurement, which could harm financial stability and reduce trust among stakeholders.  
Choosing between straightforward, interpretable models like Logistic Regression with Weight of Evidence (WoE) and more complex, high-performing models like Gradient Boosting involves significant trade-offs in a regulated financial environment. Simple models provide transparency and help with compliance with Basel II’s demands for clear risk factor explanations. This makes them suitable for regulatory audits and communication with stakeholders. However, they might compromise predictive accuracy, possibly underestimating risk and impacting loan pricing or capital distribution. On the other hand, complex models like Gradient Boosting deliver better predictive performance. They capture complicated risk patterns, but their "black-box" characteristics make regulatory justification harder and raise the risk of non-compliance. Finding a balance between clarity and performance is crucial to meet Basel II’s standards while improving risk management and business results.


## EDA(exploratory data analysis)

### Dataset Summary
- **Rows and Columns**: The dataset contains 95,662 rows and 16 columns.
- **Data Types**: 
  - 1 float column
  - 11 object columns
  - 4 integer columns
- **Country Information**: The dataset is sourced from a single country code.

### Key Observations
- **Distribution**:
  - The features `Amount`, `Value`, and `FraudResult` are skewed to the left.
  - Box plots indicate:
    - Limited variability in `CountryCode` and `PricingStrategy`.
    - Skewed distributions with outliers in `Amount` and `Value`.
    - Highly skewed `FraudResult` with rare positive cases.
- **Correlations**:
  - `CountryCode` and `Value`: Very strong positive correlation (0.99), suggesting near-perfect alignment.
  - `Amount` and `Value`: Very strong positive correlation (0.99), indicating they move almost identically.
  - `CountryCode`, `Amount`, and `Value` with `PricingStrategy`: Weak negative correlations (-0.06, -0.02), showing little to no linear relationship.
  - `CountryCode`, `Amount`, and `Value` with `FraudResult`: Moderate positive correlations (0.56, 0.57), suggesting some association with `FraudResult`.
  - `PricingStrategy` with `FraudResult`: Very weak negative correlation (-0.03), indicating almost no relationship.
  - All variables have a perfect correlation (1.00) with themselves, as expected.