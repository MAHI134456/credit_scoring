import requests

url = "http://127.0.0.1:8000/predict"
data = {
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
}
response = requests.post(url, json=data)
print(response.status_code)
print(response.json())