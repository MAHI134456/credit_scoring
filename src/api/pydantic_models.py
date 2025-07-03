"""from pydantic import BaseModel

class CustomerData(BaseModel):
    CurrencyCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Recency: float
    Frequency: float
    Monetary: float

class PredictionResponse(BaseModel):
    risk_probability: float
"""


from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str  # Expect ISO format, e.g., "2023-01-01 12:00:00"
    PricingStrategy: int
    FraudResult: Optional[int] = None  # Optional for prediction

    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "TXN123",
                "BatchId": "BATCH001",
                "AccountId": "ACC001",
                "SubscriptionId": "SUB001",
                "CustomerId": "CustomerId_4406",
                "CurrencyCode": "USD",
                "CountryCode": 255,
                "ProviderId": "PROV001",
                "ProductId": "PROD001",
                "ProductCategory": "Finance",
                "ChannelId": "WEB",
                "Amount": 150.50,
                "Value": 150,
                "TransactionStartTime": "2023-01-01 12:00:00",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        }

class PredictionResponse(BaseModel):
    risk_probability: float

    class Config:
        schema_extra = {
            "example": {
                "risk_probability": 0.75
            }
        }