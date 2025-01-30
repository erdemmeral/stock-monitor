import json
from multiprocessing import Pool
import multiprocessing
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import joblib
import logging
from pathlib import Path
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertModel,
    AutoTokenizer,
    AutoModel
)
import numpy as np
import signal
import functools
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import scipy.sparse
import aiohttp
import uuid
from collections import defaultdict

from requests import Session

# Create logs directory if it doesn't exist
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a unique log file name with timestamp
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will continue to print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting new training session. Logging to: {log_filename}")
# Configure logger at the top of the file

# Then replace the print statements with logger calls
class NewsSemanticAnalyzer:
    def __init__(self):
        # Initialize BERT model for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()

        # Store historical patterns
        self.news_embeddings = []  # Store embeddings
        self.price_impacts = []    # Store corresponding price impacts
        self.news_clusters = defaultdict(list)  # Store clustered news
        self.cluster_impacts = defaultdict(list)  # Store impact patterns per cluster

    def get_embedding(self, text):
        """Generate embedding for text using BERT"""
        try:
            # Tokenize and get BERT embedding
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get text embedding
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            return embedding[0].numpy()

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def find_similar_news(self, embedding, threshold=0.8):
        """Find similar historical news articles based on embedding similarity"""
        if not self.news_embeddings:
            return []

        # Calculate cosine similarity with all stored embeddings
        similarities = cosine_similarity([embedding], self.news_embeddings)[0]

        # Get indices of similar news
        similar_indices = np.where(similarities >= threshold)[0]

        return [(idx, similarities[idx]) for idx in similar_indices]

    def cluster_news(self, embeddings, eps=0.3):
        """Cluster news articles using DBSCAN"""
        if len(embeddings) < 2:
            return [-1] * len(embeddings)

        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(embeddings)
        return clustering.labels_

    def analyze_cluster_impact(self, cluster_id, timeframe):
        """Analyze historical price impact patterns for a cluster"""
        if cluster_id not in self.cluster_impacts:
            return None

        impacts = self.cluster_impacts[cluster_id]
        if not impacts:
            return None

        # Calculate statistics
        impacts_array = np.array([impact[timeframe] for impact in impacts if timeframe in impact])
        if len(impacts_array) == 0:
            return None

        return {
            'mean': np.mean(impacts_array),
            'std': np.std(impacts_array),
            'median': np.median(impacts_array),
            'count': len(impacts_array)
        }

    def update_patterns(self, text, price_impacts):
        """Update historical patterns with new data"""
        try:
            embedding = self.get_embedding(text)
            if embedding is None:
                logger.warning("Failed to generate embedding for text")
                return

            # Store new data
            self.news_embeddings.append(embedding)
            self.price_impacts.append(price_impacts)
            
            logger.info(f"Added new pattern. Total patterns: {len(self.news_embeddings)}")

            # Recluster if we have enough data
            if len(self.news_embeddings) >= 10:
                logger.info("Reclustering patterns...")
                clusters = self.cluster_news(np.array(self.news_embeddings))

                # Reset clusters
                self.news_clusters.clear()
                self.cluster_impacts.clear()

                # Update clusters and their impacts
                for idx, cluster_id in enumerate(clusters):
                    if cluster_id != -1:  # Not noise
                        self.news_clusters[cluster_id].append(idx)
                        self.cluster_impacts[cluster_id].append(self.price_impacts[idx])
                
                logger.info(f"Created {len(self.news_clusters)} clusters")
                for cluster_id, indices in self.news_clusters.items():
                    logger.info(f"Cluster {cluster_id}: {len(indices)} articles")

        except Exception as e:
            logger.error(f"Error updating patterns: {str(e)}")
            logger.exception("Full traceback:")

    def get_semantic_impact_prediction(self, text, timeframe):
        """Predict price impact based on semantic similarity to historical patterns"""
        embedding = self.get_embedding(text)
        if embedding is None:
            return None

        similar_news = self.find_similar_news(embedding)
        if not similar_news:
            return None

        # Weight predictions by similarity
        weighted_impacts = []
        total_weight = 0

        for idx, similarity in similar_news:
            if timeframe in self.price_impacts[idx]:
                impact = self.price_impacts[idx][timeframe]
                weight = similarity
                weighted_impacts.append(impact * weight)
                total_weight += weight

        if not weighted_impacts:
            return None

        # Calculate weighted average impact
        predicted_impact = sum(weighted_impacts) / total_weight

        return predicted_impact

class FinBERTSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model.eval()  # Set the model to evaluation mode
        
    def analyze_sentiment(self, text):
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
            # Convert predictions to probabilities
            probs = probabilities[0].tolist()
            
            # Calculate continuous sentiment score
            # Map positive to 1, negative to -1, neutral to 0 and take weighted average
            sentiment_score = probs[0] - probs[1]  # positive - negative
            
            # Calculate confidence score as the maximum of positive and negative probabilities
            confidence_score = max(probs[0], probs[1])
            
            return {
                'score': sentiment_score,  # Continuous score between -1 and 1
                'confidence': confidence_score,  # Confidence score between 0 and 1
                'probabilities': {
                    'positive': probs[0],
                    'negative': probs[1],
                    'neutral': probs[2]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

class MarketMLTrainer:
    def __init__(self):
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.semantic_analyzer = NewsSemanticAnalyzer()
        # Initialize TF-IDF vectorizer without limiting features
        self.vectorizer = TfidfVectorizer(
            min_df=5,           # Minimum document frequency
            max_df=0.95,        # Maximum document frequency
            stop_words='english',
        )
        self.symbols = self.get_symbols()
        # Initialize models to None, will be properly initialized after feature creation
        self.models = {
            '1h': None,
            '1wk': None,
            '1mo': None
        }
        
        # Store semantic patterns
        self.semantic_patterns = {
            '1h': defaultdict(list),
            '1wk': defaultdict(list),
            '1mo': defaultdict(list)
        }
        self.thresholds = {
            '1h': 5.0,   # 5% threshold
            '1wk': 10.0,  # 10% threshold
            '1mo': 20.0   # 20% threshold
        }
        self.significance_criteria = {
            '1h': {
                'threshold': 5.0,
                'min_volume': 100000,
                'min_samples': 3
            },
            '1wk': {
                'threshold': 10.0,
                'min_volume': 100000,
                'min_samples': 5,
                'consistency': 0.7
            },
            '1mo': {
                'threshold': 20.0,
                'min_volume': 100000,
                'min_samples': 20,
                'consistency': 0.7
            }
        }

    def get_symbols(self):
        try:
            with open('./stock_tickers.txt', 'r') as file:
                symbols = file.read().strip().split('\n')
            return symbols
        except Exception as e:
            print(f"Error reading symbols file: {e}")
        return []
    
    def get_full_article_text(self,url):
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get domain to handle different sites
            domain = urlparse(url).netloc
            
            # Different parsing rules for different sites
            if 'yahoo.com' in domain:
                # Yahoo Finance articles
                article_content = soup.find('div', {'class': 'caas-body'})
                if article_content:
                    return article_content.get_text(separator=' ', strip=True)
            elif 'seekingalpha.com' in domain:
                # Seeking Alpha articles
                article_content = soup.find('div', {'data-test-id': 'article-content'})
                if article_content:
                    return article_content.get_text(separator=' ', strip=True)
            elif 'reuters.com' in domain:
                # Reuters articles
                article_content = soup.find('div', {'class': 'article-body'})
                if article_content:
                    return article_content.get_text(separator=' ', strip=True)
            
            # Generic article content extraction
            # Look for common article container classes/IDs
            possible_content = soup.find(['article', 'main', 'div'], 
                                    {'class': ['article', 'content', 'article-content', 'story-content']})
            if possible_content:
                return possible_content.get_text(separator=' ', strip=True)
                
            return None
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
            return None
    def process_symbol(self, symbol):
        """Process a single symbol and collect training data"""
        try:
            # Create a local semantic analyzer for this process
            local_semantic_analyzer = NewsSemanticAnalyzer()
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return []
                
            symbol_data = []
            processed_articles = 0
            max_articles = 10
            
            logger.info(f"Processing {len(news)} news articles for {symbol}")
            
            for article in news:
                if processed_articles >= max_articles:
                    break
                    
                try:
                    # Extract article details
                    title = article.get('title', '')
                    link = article.get('link', '')
                    publish_time = article.get('providerPublishTime')
                    
                    if not all([title, link, publish_time]):
                        continue
                    
                    # Get full article text
                    full_text = self.get_full_article_text(link) or ''
                    content = title + ' ' + full_text
                    
                    if len(content.strip()) < 10:
                        continue
                    
                    # Analyze sentiment
                    sentiment = self.finbert_analyzer.analyze_sentiment(content)
                    if not sentiment:
                        continue
                    
                    # Analyze stock price changes
                    changes = self.analyze_stock(stock, publish_time)
                    if not changes:
                        continue
                    
                    # Create sample with semantic data
                    embedding = local_semantic_analyzer.get_embedding(content)
                    if embedding is not None:
                        sample = {
                            'text': content,
                            'changes': changes,
                            'symbol': symbol,
                            'date': datetime.fromtimestamp(publish_time),
                            'sentiment': sentiment,
                            'embedding': embedding  # Store embedding with the sample
                        }
                        symbol_data.append(sample)
                        processed_articles += 1
                        logger.info(f"Processed article {processed_articles} for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(symbol_data)} articles for {symbol}")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return []

    def analyze_stock(self, stock, publish_time):
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning, 
                          message='The \'unit\' keyword in TimedeltaIndex construction is deprecated')
        changes = {}
        
        # Convert publish_time from timestamp to datetime
        publish_date = datetime.fromtimestamp(publish_time)
        
        timeframes = {
            '1h': {'interval': '1h', 'days_before': 2, 'days_after': 2},
            '1wk': {'interval': '1d', 'days_before': 10, 'days_after': 10},
            '1mo': {'interval': '1d', 'days_before': 35, 'days_after': 35}
        }
        
        for timeframe, params in timeframes.items():
            try:
                # Get price data
                end_date = publish_date + timedelta(days=params['days_after'])
                start_date = publish_date - timedelta(days=params['days_before'])
                
                # Get historical data with a single call
                period_prices = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=params['interval'],
                    prepost=False  # Only regular market hours
                )
                
                if len(period_prices) < 2:
                    continue

                # Calculate percentage change
                start_price = period_prices['Close'].iloc[0]
                end_price = period_prices['Close'].iloc[-1]
                
                change = ((end_price - start_price) / start_price) * 100
                if not pd.isna(change):  # Only add valid changes
                    changes[timeframe] = change
                    if abs(change) > 1.0:  # Only log significant changes
                        logger.info(f"{stock.ticker}: {timeframe} change of {change:.2f}%")
                    
            except Exception as e:
                logger.error(f"Error processing {stock.ticker} for {timeframe}: {str(e)}")
                    
        return changes

    def check_trend_consistency(self, prices, timeframe):
        """Check if price movement is consistent"""
        min_samples = self.significance_criteria[timeframe]['min_samples']
        
        if len(prices) < min_samples:
            logger.info(f"Insufficient samples for {timeframe}: {len(prices)} vs required {min_samples}")
            return False
            
        # Calculate price changes between consecutive points
        changes = prices['Close'].pct_change().dropna()
        
        if changes.empty:
            return False
            
        # Count consistent moves (same direction as overall trend)
        overall_direction = np.sign(changes.mean())
        consistent_moves = np.sum(np.sign(changes) == overall_direction)
        consistency_ratio = consistent_moves / len(changes)
        
        required_consistency = self.significance_criteria[timeframe].get('consistency', 0.0)
        is_consistent = consistency_ratio >= required_consistency
        
        logger.info(f"Trend consistency for {timeframe}: {consistency_ratio:.2f} vs required {required_consistency} - {'OK' if is_consistent else 'X'}")
        
        return is_consistent

    def calculate_normalized_score(self, changes):
        normalized_scores = {}
        for timeframe, change in changes.items():
            if change is not None:  # Add this check
                threshold = self.thresholds[timeframe]
                normalized_score = change / threshold
                normalized_scores[timeframe] = normalized_score
                print(f"{timeframe} Change: {change:.2f}% (Normalized Score: {normalized_score:.2f})")
        return normalized_scores
    
    def calculate_article_impact(self, sentiment_score, price_changes, timeframes=['1h', '1wk', '1mo']):
        """Calculate the impact score of an article based on sentiment and actual price movements"""
        impact_scores = {}
        
        for timeframe in timeframes:
            if timeframe not in price_changes:
                continue
                
            price_change = price_changes[timeframe]
            threshold = self.thresholds[timeframe]
            
            # Check if price change exceeds threshold
            if abs(price_change) >= threshold:
                # Calculate direction alignment
                sentiment_direction = np.sign(sentiment_score)
                price_direction = np.sign(price_change)
                direction_match = sentiment_direction == price_direction
                
                # Calculate impact score
                # Higher score if:
                # 1. Price change is larger relative to threshold
                # 2. Sentiment correctly predicted direction
                # 3. Shorter timeframe (more immediate impact)
                timeframe_weight = {
                    '1h': 1.0,
                    '1wk': 0.8,
                    '1mo': 0.6
                }
                
                impact = (abs(price_change) / threshold) * timeframe_weight[timeframe]
                if direction_match:
                    impact *= 1.5  # Boost score for correct predictions
                else:
                    impact *= 0.5  # Reduce score for incorrect predictions
                
                impact_scores[timeframe] = impact
            else:
                impact_scores[timeframe] = 0.0
                
        return impact_scores

    def prepare_training_sample(self, sample, impact_scores):
        """Prepare a training sample with adjusted features based on historical impact"""
        try:
            # Get base TF-IDF features
            sample_tfidf = self.vectorizer.transform([sample['text']])
            logger.info(f"TF-IDF features shape: {sample_tfidf.shape}")
            
            # Get sentiment features
            sentiment = sample.get('sentiment', {})
            if not sentiment:
                return None, None
            
            # Calculate average impact across timeframes
            avg_impact = np.mean(list(impact_scores.values()))
            
            # Ensure sentiment features are 2D
            sentiment_features = np.array([
                sentiment['score'] * (1 + avg_impact),  # Amplify sentiment if historically impactful
                sentiment['confidence'],
                sentiment['probabilities']['positive'] * (1 + avg_impact),
                sentiment['probabilities']['negative'] * (1 + avg_impact),
                sentiment['probabilities']['neutral']
            ]).reshape(1, -1)  # Reshape to 2D array
            
            # Convert to sparse matrix with proper shape
            sentiment_sparse = scipy.sparse.csr_matrix(sentiment_features)
            logger.info(f"Sentiment features shape: {sentiment_sparse.shape}")
            
            # Combine features ensuring both are 2D
            combined_features = scipy.sparse.hstack([sample_tfidf, sentiment_sparse])
            logger.info(f"Combined features shape: {combined_features.shape}")
            
            return combined_features, avg_impact
            
        except Exception as e:
            logger.error(f"Error preparing training sample: {str(e)}")
            return None, None

    def calculate_impact(self, prices, publish_time):
        """Calculate price impact with timezone-aware datetime handling"""
        if prices.empty:
            return {'impact': 1, 'scores': None, 'changes': None}

        # Ensure publish_time is timezone-aware
        publish_datetime = datetime.fromtimestamp(publish_time, tz=pytz.UTC)
        logger.info(f"Publish Date: {publish_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        changes = {}
        start_price = None
        
        # Convert index to UTC if not already
        if prices.index.tz is None:
            prices.index = prices.index.tz_localize(pytz.UTC)
            
        for idx, row in prices.iterrows():
            if idx >= publish_datetime:
                start_price = row['Close']
                break

        if start_price is None:
            return {'impact': 1, 'scores': None, 'changes': None}

        # Calculate changes for each timeframe
        timeframes = {
            '1h': timedelta(hours=1),
            '1w': timedelta(days=7),
            '1m': timedelta(days=30)
        }

        for timeframe, delta in timeframes.items():
            end_time = publish_datetime + delta
            period_prices = prices[prices.index <= end_time]
            if not period_prices.empty:
                end_price = period_prices['Close'].iloc[-1]
                changes[timeframe] = ((end_price - start_price) / start_price) * 100

        # Calculate normalized scores
        scores = self.calculate_normalized_score(changes)

        # Determine overall impact based on best normalized score
        best_score = max(scores.values(), default=0)
        if best_score >= 1.0:  # Met or exceeded threshold
            impact = 2
        elif best_score <= -1.0:  # Met or exceeded negative threshold
            impact = 0
        else:
            impact = 1

        return {
            'impact': impact,
            'scores': scores,
            'changes': changes
        }

    def train_with_impact_scores(self, training_data, timeframe):
        """Train model with impact-adjusted samples ensuring proper feature dimensionality"""
        try:
            X_features_list = []
            y = []
            sample_weights = []
            
            logger.info(f"Processing {len(training_data)} samples for {timeframe} model")
            
            for sample in training_data:
                # Calculate impact scores for this sample
                impact_scores = self.calculate_article_impact(
                    sample['sentiment']['score'],
                    sample['changes']
                )
                
                # Prepare features with impact adjustment
                features, impact = self.prepare_training_sample(sample, impact_scores)
                if features is None:
                    continue
                
                # Ensure features are 2D
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                X_features_list.append(features)
                
                # Get the change value for this timeframe
                change = sample['changes'].get(timeframe)
                if change is None:
                    logger.warning(f"No change value found for timeframe {timeframe}")
                    continue
                    
                y.append(change)
                
                # Use impact as sample weight
                weight = 1.0 + impact_scores.get(timeframe, 0)
                sample_weights.append(weight)
            
            if not X_features_list:
                logger.error(f"No valid samples collected for {timeframe} model")
                return False
                
            logger.info(f"Collected {len(X_features_list)} valid samples for {timeframe} model")
                
            # Combine features ensuring proper dimensionality
            X_combined = scipy.sparse.vstack(X_features_list)
            y_array = np.array(y, dtype=float)
            weights_array = np.array(sample_weights)
            
            # Remove NaN values
            valid_mask = ~np.isnan(y_array)
            X_clean = X_combined[valid_mask]
            y_clean = y_array[valid_mask]
            weights_clean = weights_array[valid_mask]
            
            logger.info(f"Final feature matrix shape for {timeframe}: {X_clean.shape}")
            logger.info(f"Final target vector shape for {timeframe}: {y_clean.shape}")
            
            # Initialize the model with the correct number of features
            if self.models[timeframe] is None:
                n_features = X_clean.shape[1]
                logger.info(f"Initializing {timeframe} model with {n_features} features")
                self.models[timeframe] = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=0.01,
                    max_iter=1000,
                    tol=1e-3,
                    early_stopping=True,
                    warm_start=True  # Allow incremental learning
                )
            
            # Train the model
            self.models[timeframe].fit(X_clean, y_clean, sample_weight=weights_clean)
            logger.info(f"Successfully trained {timeframe} model")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {timeframe}: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def collect_and_train(self, symbols):
        """Collect news data and train models."""
        logger.info("Starting training process...")
        training_data = []
        failed_stocks = set()

        # Configure yfinance cache
        cache_dir = Path("./cache/yfinance")
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        
        # Process only first 100 stocks for testing
        test_symbols = symbols[:100]
        chunk_size = 20  # Process 20 symbols at a time
        num_processes = 4  # Use 4 processes

        # First pass: Process test stocks in chunks
        logger.info("PHASE 1: Processing first 100 stocks in chunks")
        logger.info(f"Selected stocks: {', '.join(test_symbols[:10])}...")
        for i in range(0, len(test_symbols), chunk_size):
            chunk = test_symbols[i:i+chunk_size]
            chunk_num = i//chunk_size + 1
            total_chunks = len(symbols)//chunk_size + 1
            logger.info(f"\nProcessing chunk {chunk_num}/{total_chunks}")
            logger.info(f"Symbols in chunk: {', '.join(chunk)}")

            with Pool(processes=num_processes) as pool:
                results = []
                for symbol, result in zip(chunk, pool.imap_unordered(self.process_symbol, chunk)):
                    if result and len(result) > 0:
                        results.extend(result)
                        logger.info(f"Successfully processed {symbol} with {len(result)} samples")
                    else:
                        failed_stocks.add(symbol)
                        logger.info(f"Failed to process {symbol}")

                training_data.extend(results)
                logger.info(f"Total samples collected so far: {len(training_data)}")
                logger.info(f"Failed stocks so far: {len(failed_stocks)}")

            # Update semantic patterns after each chunk
            self.update_semantic_patterns(training_data)

            # Add delay between chunks
            if i + chunk_size < len(symbols):
                logger.info("Taking a break between chunks...")
                time.sleep(10)

        if not training_data:
            logger.error("No training data collected!")
            return None

        logger.info("\nPreparing to train models...")
        logger.info(f"Total training samples: {len(training_data)}")
        logger.info(f"Total failed stocks: {len(failed_stocks)}")

        try:
            # Fit the vectorizer first without limiting features
            logger.info("Fitting TF-IDF vectorizer...")
            X_text = [sample['text'] for sample in training_data]
            self.vectorizer.fit(X_text)
            vocab_size = len(self.vectorizer.vocabulary_)
            logger.info(f"Vectorizer fitted successfully. Vocabulary size: {vocab_size}")

            # Initialize models with correct feature dimensions
            feature_sample = self.prepare_training_sample(training_data[0], {})
            if feature_sample[0] is not None:
                n_features = feature_sample[0].shape[1]
                logger.info(f"Initializing models with {n_features} features")
                
                for timeframe in ['1h', '1wk', '1mo']:
                    self.models[timeframe] = SGDRegressor(
                        learning_rate='adaptive',
                        eta0=0.01,
                        max_iter=1000,
                        tol=1e-3,
                        early_stopping=True,
                        warm_start=True  # Allow incremental learning
                    )

            # Train models for each timeframe
            for timeframe in ['1h', '1wk', '1mo']:
                logger.info(f"\nTraining {timeframe} model...")
                success = self.train_with_impact_scores(training_data, timeframe)
                if success:
                    logger.info(f"Successfully trained {timeframe} model")
                else:
                    logger.error(f"Failed to train {timeframe} model")

            return self.models

        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
            return None

    def update_semantic_patterns(self, samples):
        """Update semantic patterns after collecting samples"""
        try:
            # Reset semantic analyzer
            self.semantic_analyzer = NewsSemanticAnalyzer()
            
            # Add all patterns
            for sample in samples:
                if 'embedding' in sample:
                    self.semantic_analyzer.news_embeddings.append(sample['embedding'])
                    self.semantic_analyzer.price_impacts.append(sample['changes'])
            
            # Perform clustering if we have enough samples
            if len(self.semantic_analyzer.news_embeddings) >= 10:
                logger.info(f"Clustering {len(self.semantic_analyzer.news_embeddings)} patterns...")
                clusters = self.semantic_analyzer.cluster_news(np.array(self.semantic_analyzer.news_embeddings))
                
                # Update clusters
                self.semantic_analyzer.news_clusters.clear()
                self.semantic_analyzer.cluster_impacts.clear()
                
                for idx, cluster_id in enumerate(clusters):
                    if cluster_id != -1:  # Not noise
                        self.semantic_analyzer.news_clusters[cluster_id].append(idx)
                        self.semantic_analyzer.cluster_impacts[cluster_id].append(self.semantic_analyzer.price_impacts[idx])
                
                logger.info(f"Created {len(self.semantic_analyzer.news_clusters)} clusters")
                for cluster_id, indices in self.semantic_analyzer.news_clusters.items():
                    logger.info(f"Cluster {cluster_id}: {len(indices)} articles")
                    
        except Exception as e:
            logger.error(f"Error updating semantic patterns: {str(e)}")
            logger.exception("Full traceback:")

    def _get_sentiment_multiplier(self, sentiment):
        """
        Adjust price change based on continuous sentiment score
        
        :param sentiment: Dictionary containing sentiment analysis results
        :return: Multiplier for price change
        """
        # Get the continuous score and probabilities
        score = sentiment['score']  # Between -1 and 1
        confidence = sentiment['confidence']  # Max of positive/negative probabilities
        neutral_prob = sentiment['probabilities']['neutral']
        
        # Calculate neutral dampener (reduce effect when neutral probability is high)
        neutral_dampener = 1.0 - neutral_prob
        
        # Calculate base multiplier from continuous score
        # Maps [-1, 1] to [0.7, 1.3]
        base_multiplier = 1.0 + (score * 0.3)
        
        # Adjust multiplier with confidence and neutral dampener
        adjusted_multiplier = 1.0 + (base_multiplier - 1.0) * confidence * neutral_dampener
        
        return adjusted_multiplier
    
    def predict(self, text, timeframe):
        """
        Make a prediction with sentiment and semantic integration

        :param text: Input text
        :param timeframe: Prediction timeframe
        :return: Predicted price change
        """
        # Vectorize text
        X_tfidf = self.vectorizer.transform([text])

        # Analyze sentiment
        sentiment = self.finbert_analyzer.analyze_sentiment(text)

        # Get semantic prediction
        semantic_pred = self.semantic_analyzer.get_semantic_impact_prediction(text, timeframe)

        # Get base prediction from ML model
        base_pred = self.models[timeframe].predict(X_tfidf)[0]

        # Get sentiment multiplier
        sentiment_multiplier = self._get_sentiment_multiplier(sentiment) if sentiment else 1.0

        # Combine predictions
        if semantic_pred is not None:
            # Weight the predictions:
            # - Base ML prediction: 40%
            # - Semantic prediction: 40%
            # - Sentiment adjustment: 20%
            weighted_pred = (
                0.4 * base_pred +
                0.4 * semantic_pred +
                0.2 * (base_pred * sentiment_multiplier)
            )
        else:
            # If no semantic prediction, fall back to sentiment-adjusted base prediction
            weighted_pred = base_pred * sentiment_multiplier

        # Log detailed prediction breakdown
        logger.info(f"\nPrediction Breakdown for {timeframe}:")
        logger.info(f"1. Base ML Prediction: {base_pred:.2f}%")
        if sentiment:
            logger.info(f"2. Sentiment Score: {sentiment['score']:.2f}")
            logger.info(f"   Sentiment Multiplier: {sentiment_multiplier:.2f}")
        if semantic_pred is not None:
            logger.info(f"3. Semantic Prediction: {semantic_pred:.2f}%")

            # Find and log similar historical patterns
            embedding = self.semantic_analyzer.get_embedding(text)
            if embedding is not None:
                similar_news = self.semantic_analyzer.find_similar_news(embedding)
                if similar_news:
                    logger.info("\nSimilar Historical Patterns:")
                    for idx, similarity in similar_news[:3]:  # Show top 3
                        impact = self.semantic_analyzer.price_impacts[idx].get(timeframe)
                        if impact is not None:
                            logger.info(f"- Similarity: {similarity:.2f}, Impact: {impact:.2f}%")

        logger.info(f"\nFinal Weighted Prediction: {weighted_pred:.2f}%")

        return weighted_pred
    
    def save_models(self):
        os.makedirs('app/models', exist_ok=True)

        # Save each timeframe model
        for timeframe, model in self.models.items():
            model_path = f'app/models/market_model_{timeframe}.joblib'
            joblib.dump(model, model_path)
            logger.info(f"Saved {timeframe} model to {model_path}")

        # Save vectorizer
        vectorizer_path = 'app/models/vectorizer.joblib'
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Saved vectorizer to {vectorizer_path}")

        # Save semantic patterns
        semantic_data = {
            'embeddings': self.semantic_analyzer.news_embeddings,
            'price_impacts': self.semantic_analyzer.price_impacts,
            'clusters': dict(self.semantic_analyzer.news_clusters),
            'cluster_impacts': dict(self.semantic_analyzer.cluster_impacts)
        }
        semantic_path = 'app/models/semantic_patterns.joblib'
        joblib.dump(semantic_data, semantic_path)
        logger.info(f"Saved semantic patterns to {semantic_path}")

    def load_models(self):
        """Load trained models and patterns"""
        try:
            # Load semantic patterns if they exist
            semantic_path = 'app/models/semantic_patterns.joblib'
            if os.path.exists(semantic_path):
                semantic_data = joblib.load(semantic_path)
                self.semantic_analyzer.news_embeddings = semantic_data['embeddings']
                self.semantic_analyzer.price_impacts = semantic_data['price_impacts']
                self.semantic_analyzer.news_clusters = defaultdict(list, semantic_data['clusters'])
                self.semantic_analyzer.cluster_impacts = defaultdict(list, semantic_data['cluster_impacts'])
                logger.info("Loaded semantic patterns successfully")

            # Load models for each timeframe
            for timeframe in self.models.keys():
                model_path = f'app/models/market_model_{timeframe}.joblib'
                if os.path.exists(model_path):
                    self.models[timeframe] = joblib.load(model_path)
                    logger.info(f"Loaded {timeframe} model successfully")

            # Load vectorizer
            vectorizer_path = 'app/models/vectorizer.joblib'
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("Loaded vectorizer successfully")

            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

class PortfolioTrackerService:
    def __init__(self):
        self.base_url = "https://portfolio-tracker-rough-dawn-5271.fly.dev"
        self.session = None
        self.active_transactions = {}  # Store active transaction IDs by symbol
        
    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float):
        """Send buy signal to portfolio tracker with a transaction ID"""
        await self._ensure_session()
        
        # Generate a unique transaction ID
        transaction_id = str(uuid.uuid4())
        self.active_transactions[symbol] = transaction_id
        
        entry_date = datetime.now()
        target_date = entry_date + timedelta(days=30)
        
        data = {
            "transactionId": transaction_id,  # Add transaction ID
            "symbol": symbol,
            "entryPrice": entry_price,
            "targetPrice": target_price,
            "entryDate": entry_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "targetDate": target_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/signals/buy", json=data) as response:
                if response.status == 200:
                    logger.info(f"Buy signal sent for {symbol} (Transaction ID: {transaction_id})")
                    return await response.json()
                else:
                    logger.error(f"Failed to send buy signal for {symbol}")
                    self.active_transactions.pop(symbol, None)  # Remove on failure
                    return None
        except Exception as e:
            logger.error(f"Error sending buy signal for {symbol}: {str(e)}")
            self.active_transactions.pop(symbol, None)  # Remove on error
            return None
            
    async def send_sell_signal(self, symbol: str, selling_price: float):
        """Send sell signal to portfolio tracker with matching transaction ID"""
        await self._ensure_session()
        
        # Get the transaction ID for this symbol
        transaction_id = self.active_transactions.get(symbol)
        if not transaction_id:
            logger.error(f"No active transaction found for {symbol}")
            return None
            
        selling_date = datetime.now()
        
        data = {
            "transactionId": transaction_id,  # Include the same transaction ID
            "symbol": symbol,
            "sellingPrice": selling_price,
            "sellingDate": selling_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/signals/sell", json=data) as response:
                if response.status == 200:
                    logger.info(f"Sell signal sent for {symbol} (Transaction ID: {transaction_id})")
                    # Remove the completed transaction
                    self.active_transactions.pop(symbol, None)
                    return await response.json()
                else:
                    logger.error(f"Failed to send sell signal for {symbol}")
                    return None
        except Exception as e:
            logger.error(f"Error sending sell signal for {symbol}: {str(e)}")
            return None

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Market ML Training...")
    trainer = MarketMLTrainer()

    # Get all symbols
    symbols = trainer.get_symbols()
    logger.info(f"Total symbols to process: {len(symbols)}")

    try:
        # Create models directory if it doesn't exist
        os.makedirs('app/models', exist_ok=True)

        # Try to load existing models and patterns
        logger.info("Checking for existing models and patterns...")
        if trainer.load_models():
            logger.info("Successfully loaded existing models and patterns")
            logger.info("Will update existing patterns with new data")
        else:
            logger.info("No existing models found or loading failed")
            logger.info("Will train new models from scratch")

        # Train/update models
        logger.info("Starting training process...")
        models = trainer.collect_and_train(symbols)

        if models:
            logger.info("Training completed successfully")
            trainer.save_models()
            logger.info("Models and patterns saved successfully")
            logger.info("Saved files:")
            logger.info("- vectorizer.joblib")
            logger.info("- market_model_1h.joblib")
            logger.info("- market_model_1wk.joblib")
            logger.info("- market_model_1mo.joblib")
            logger.info("- semantic_patterns.joblib")

            # Log semantic analysis statistics
            num_patterns = len(trainer.semantic_analyzer.news_embeddings)
            num_clusters = len(trainer.semantic_analyzer.news_clusters)
            logger.info(f"\nSemantic Analysis Statistics:")
            logger.info(f"Total patterns collected: {num_patterns}")
            logger.info(f"Number of news clusters: {num_clusters}")

            # Log cluster statistics
            if num_clusters > 0:
                logger.info("\nCluster Statistics:")
                for cluster_id in trainer.semantic_analyzer.news_clusters.keys():
                    cluster_size = len(trainer.semantic_analyzer.news_clusters[cluster_id])
                    impacts = trainer.semantic_analyzer.cluster_impacts[cluster_id]
                    avg_impact = np.mean([impact.get('1mo', 0) for impact in impacts])
                    logger.info(f"Cluster {cluster_id}: {cluster_size} articles, Avg 1mo impact: {avg_impact:.2f}%")
        else:
            logger.error("Training failed - no models were produced")

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.exception("Full error traceback:")

if __name__ == "__main__":
    main()