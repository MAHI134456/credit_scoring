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

### Next Steps
- Enhance the model with additional features (e.g., time-based features from TransactionStartTime).
- Address multicollinearity between `Amount` and `Value`.
- Improve dashboard visualization for end-users.

## Proxy Target Variable Engineering (Task 4)
### Objective
Since no pre-existing "credit risk" column was available, a proxy target variable (`is_high_risk`) was engineered to identify high-risk customers based on their transaction behavior.

### Methodology
1. **Calculate RFM Metrics**:
   - **Recency**: Calculated as the number of days since the last transaction, using a snapshot date (e.g., the latest `TransactionStartTime` in the dataset, adjusted to a future date like 2018-12-01 for consistency).
   - **Frequency**: Count of transactions per `CustomerId`.
   - **Monetary**: Sum of `Amount` (or `Value`) per `CustomerId`.
   - RFM values were computed for each `CustomerId` from the transaction history.

2. **Cluster Customers**:
   - Preprocessed RFM features by scaling them using StandardScaler to ensure meaningful clustering.
   - Applied K-Means clustering with 3 clusters and `random_state=42` for reproducibility.
   - Segmented customers based on their RFM profiles.

3. **Define and Assign "High-Risk" Label**:
   - Analyzed the clusters to identify the high-risk segment, characterized by low Frequency and low Monetary value (indicating disengaged customers with a high likelihood of default).
   - Assigned `is_high_risk = 1` to customers in the high-risk cluster and `is_high_risk = 0` to all others.
   - Created a new binary target column `is_high_risk` in the dataset.

4. **Integrate the Target Variable**:
   - Merged the `is_high_risk` column back into the main processed dataset, making it available for model training.

## Model Training
The best model, `credit_scoring_model`, was trained using a dataset preprocessed into 51 features, including the engineered `is_high_risk` target variable, and registered with the `production` alias in the MLflow registry. The model predicts the risk probability based on transaction data.

## Model Deployment
- **API**: A FastAPI application is deployed at `/predict`, accepting raw customer data and returning risk probability after preprocessing to match the trained model.
- **Containerization**: The service is containerized using Docker, with a `Dockerfile` and `docker-compose.yml` for easy setup.
- **Access**: Run `docker-compose up --build` to start the API on port 8000.

## CI/CD Pipeline
- **Workflow**: A GitHub Actions CI pipeline is configured in `.github/workflows/ci.yml`.
- **Triggers**: Runs on every push to the `main` branch.
- **Steps**:
  - Linting with `flake8` to ensure code quality.
  - Testing with `pytest` to validate functionality.
- **Status**: The build fails if linting or tests fail, ensuring code integrity.

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Start the API locally: `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`.
3. Test the endpoint with a sample request:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
       "TransactionId": "TransactionId_90000",
       "BatchId": "BatchId_50000",
       "AccountId": "AccountId_9999",
       "SubscriptionId": "SubscriptionId_5000",
       "CustomerId": "CustomerId_9999",
       "CurrencyCode": "UGX",
       "CountryCode": 256,
       "ProviderId": "ProviderId_6",
       "ProductId": "ProductId_10",
       "ProductCategory": "airtime",
       "ChannelId": "ChannelId_3",
       "Amount": 1500.75,
       "Value": 1500,
       "TransactionStartTime": "2018-11-15T04:00:00Z",
       "PricingStrategy": 2
   }'