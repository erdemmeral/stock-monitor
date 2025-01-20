from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import joblib

class NewsTrainer:
    def __init__(self):
        self.vectorizer = HashingVectorizer(n_features=2**16)
        self.classifier = SGDClassifier(loss='log_loss')
        
    def train_model(self, data_path: str):
        df = pd.read_csv(data_path)
        X = self.vectorizer.transform(df['news_text'])
        
        # Convert continuous scores to discrete classes
        y = np.array([0 if score <= 0.3 else 1 if score >= 0.7 else 2 for score in df['market_impact']])
        
        self.classifier.partial_fit(X, y, classes=[0, 1, 2])
        return self.classifier

def main():
    trainer = NewsTrainer()
    model = trainer.train_model('app/training/historical_news.csv')
    joblib.dump(model, 'app/training/trained_news_classifier.pkl')

if __name__ == "__main__":
    main()