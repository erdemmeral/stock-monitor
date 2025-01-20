import pandas as pd

# Training data with news and their market impact
training_data = {
    'news_text': [
        "Company reports 50% earnings beat, raises guidance",
        "FDA approves breakthrough drug treatment",
        "Major acquisition completed ahead of schedule",
        "CEO steps down amid investigation",
        "Missed quarterly earnings expectations",
        "New strategic partnership announced",
        "Stock downgraded by major analysts",
        "Successful clinical trial results",
        "Company wins government contract",
        "Layoffs announced in restructuring"
    ],
    'market_impact': [1, 1, 1, 0, 0, 0.8, 0.2, 0.9, 0.7, 0.3]
}

# Create DataFrame and save to CSV
df = pd.DataFrame(training_data)
df.to_csv('historical_news.csv', index=False)
