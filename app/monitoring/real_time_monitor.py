from urllib.parse import urlparse
from bs4 import BeautifulSoup
import numpy as np
import scipy.sparse

import requests
import telegram

# Standard library imports
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, List
import os
import logging
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import aiohttp
import yfinance as yf
import joblib
import pandas as pd
import pytz
from scipy.spatial.distance import cosine
from telegram import Update
from telegram.ext import Application, ContextTypes
from telegram.error import TelegramError
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import gc
import psutil
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sentence_transformers import SentenceTransformer

# Local application imports
import torch
from app.training.market_ml_trainer import FinBERTSentimentAnalyzer, MarketMLTrainer
from app.services.portfolio_tracker_service import PortfolioTrackerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

class HybridMarketPredictor(BaseEstimator, ClassifierMixin):
    """Hybrid model that combines TF-IDF features with semantic embeddings"""
    
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.n_features_in_ = None
        self.weights = None
        
    def fit(self, X, y):
        """Fit the model to the training data"""
        try:
            # Store number of features
            self.n_features_in_ = X.shape[1]
            
            # Convert sparse matrix to dense if needed
            if scipy.sparse.issparse(X):
                X = X.toarray()
            
            # Convert y to numpy array if needed
            y = np.array(y)
            
            # Simple linear regression for now
            # Calculate weights using normal equation: w = (X^T X)^(-1) X^T y
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            
            return self
            
        except Exception as e:
            logger.error(f"Error in HybridMarketPredictor fit: {str(e)}")
            raise
        
    def predict(self, X):
        """Make predictions on new data"""
        try:
            if self.weights is None:
                raise ValueError("Model has not been fitted yet")
                
            # Convert sparse matrix to dense if needed
            if scipy.sparse.issparse(X):
                X = X.toarray()
                
            # Check feature dimensions
            if X.shape[1] != self.n_features_in_:
                logger.warning(f"Feature dimension mismatch. Expected {self.n_features_in_}, got {X.shape[1]}")
                # Pad or truncate features to match expected dimensions
                if X.shape[1] < self.n_features_in_:
                    # Pad with zeros
                    padding = np.zeros((X.shape[0], self.n_features_in_ - X.shape[1]))
                    X = np.hstack([X, padding])
                else:
                    # Truncate
                    X = X[:, :self.n_features_in_]
            
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Make predictions
            return X_with_bias @ self.weights
            
        except Exception as e:
            logger.error(f"Error in HybridMarketPredictor predict: {str(e)}")
            raise

class ModelManager:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.semantic_patterns = None
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_analyzer = NewsSemanticAnalyzer(embedding_model=self.embedding_model)
        self.model_paths = {
            '1wk': {
                'model': 'app/models/market_model_1wk.joblib',
                'vectorizer': 'app/models/vectorizer_1wk.joblib',
                'scaler': 'app/models/scaler_1wk.joblib'
            },
            '1mo': {
                'model': 'app/models/market_model_1mo.joblib',
                'vectorizer': 'app/models/vectorizer_1mo.joblib',
                'scaler': 'app/models/scaler_1mo.joblib'
            },
            'semantic_patterns': 'app/models/semantic_patterns.joblib'
        }

    def load_models(self):
        """Load all models, vectorizers, and scalers with validation"""
        try:
            # Load models for each timeframe
            for timeframe in ['1wk', '1mo']:
                paths = self.model_paths[timeframe]
                
                # Check if all required files exist
                for key, path in paths.items():
                    if not os.path.exists(path):
                        logger.error(f"{timeframe} {key} file not found: {path}")
                        return False
                
                # Load model components
                self.models[timeframe] = joblib.load(paths['model'])
                self.vectorizers[timeframe] = joblib.load(paths['vectorizer'])
                self.scalers[timeframe] = joblib.load(paths['scaler'])
                
                # Validate components
                self.validate_model(timeframe)
                logger.info(f"Loaded and validated {timeframe} model components")
            
            # Load semantic patterns
            patterns_path = self.model_paths['semantic_patterns']
            if os.path.exists(patterns_path):
                self.semantic_patterns = joblib.load(patterns_path)
                logger.info("Loaded semantic patterns")
            else:
                logger.warning("Semantic patterns file not found")

            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def validate_model(self, timeframe):
        """Validate loaded model components"""
        model = self.models[timeframe]
        vectorizer = self.vectorizers[timeframe]
        scaler = self.scalers[timeframe]
        
        # Validate model
        if not hasattr(model, 'predict'):
            raise ValueError(f"{timeframe} model missing predict method")
            
        # Validate vectorizer
        if not hasattr(vectorizer, 'transform'):
            raise ValueError(f"{timeframe} vectorizer missing transform method")
        
        # Validate scaler
        if not hasattr(scaler, 'transform'):
            raise ValueError(f"{timeframe} scaler missing transform method")
        
        logger.info(f"Model components for {timeframe}:")
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        logger.info(f"  Scaler: {type(scaler).__name__}")

    def check_and_reload_models(self):
        """Check if models need to be reloaded and reload them if necessary"""
        try:
            current_time = datetime.now()
            should_reload = False

            # Check all model files
            for timeframe in ['1wk', '1mo']:
                for path in self.model_paths[timeframe].values():
                    if not os.path.exists(path):
                        logger.error(f"Model file not found: {path}")
                        return False
                
                mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                if current_time > mod_time:
                    should_reload = True
                    break

            if should_reload:
                logger.info("Detected model updates, reloading models...")
                return self.load_models()
            
            return True

        except Exception as e:
            logger.error(f"Error checking models: {str(e)}")
            return False

    def predict(self, text, timeframe):
        """Make a prediction using the loaded models"""
        try:
            if timeframe not in self.models:
                logger.error(f"No model loaded for timeframe: {timeframe}")
                return None
                
            # Get model components
            model = self.models[timeframe]
            vectorizer = self.vectorizers[timeframe]
            scaler = self.scalers[timeframe]
            
            # Transform text using vectorizer
            X_tfidf = vectorizer.transform([text])
            logger.debug(f"TF-IDF features shape: {X_tfidf.shape}")
            
            # Get sentiment features
            finbert = FinBERTSentimentAnalyzer()
            sentiment = finbert.analyze_sentiment(text)
            if not sentiment:
                logger.error("Failed to get sentiment analysis")
                return None
                
            # Create sentiment features array
            sentiment_features = np.array([[
                sentiment['probabilities']['positive'],
                sentiment['probabilities']['negative'],
                sentiment['probabilities']['neutral']
            ]])
            logger.debug(f"Sentiment features shape: {sentiment_features.shape}")
            
            # Combine features
            X_combined = scipy.sparse.hstack([
                X_tfidf,
                scipy.sparse.csr_matrix(sentiment_features)
            ]).tocsr()
            logger.debug(f"Combined features shape: {X_combined.shape}")
            
            # Get embedding using the shared semantic analyzer
            embedding = self.semantic_analyzer.get_embedding(text)
            if embedding is not None:
                embedding_features = np.array([embedding])
                logger.debug(f"Embedding features shape: {embedding_features.shape}")
                X_combined = scipy.sparse.hstack([
                    X_combined,
                    scipy.sparse.csr_matrix(embedding_features)
                ]).tocsr()
                logger.debug(f"Final combined features shape: {X_combined.shape}")
            
            # Make prediction
            try:
                prediction = model.predict(X_combined)[0]
                logger.debug(f"Raw prediction: {prediction}")
            except ValueError as ve:
                logger.error(f"Prediction error: {str(ve)}")
                return None
            
            # Inverse transform the prediction
            prediction_unscaled = scaler.inverse_transform([[prediction]])[0][0]
            logger.debug(f"Unscaled prediction: {prediction_unscaled}")
            
            return prediction_unscaled
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.exception("Full traceback:")
            return None

class NewsLSTM(torch.nn.Module):
    """LSTM model for news analysis"""
    def __init__(self, input_size=768, hidden_size=256, num_layers=2, dropout=0.2):
        super(NewsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, 1)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Pass through fully connected layer
        out = self.fc(last_output)
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class NewsSemanticAnalyzer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Store historical patterns
        self.news_embeddings = []
        self.price_impacts = []
        self.news_clusters = defaultdict(list)
        self.cluster_impacts = defaultdict(list)

    def get_embedding(self, text):
        """Generate embedding for text using BERT"""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return embedding[0].numpy()

    def find_similar_news(self, embedding, threshold=0.8):
        """Find similar historical news articles based on embedding similarity"""
        try:
            if not self.news_embeddings:
                return []
            similarities = cosine_similarity([embedding], self.news_embeddings)[0]
            similar_indices = np.where(similarities >= threshold)[0]
            return [(idx, similarities[idx]) for idx in similar_indices]
        except Exception as e:
            logger.error(f"Error in finding similar news: {str(e)}")
            return []

    def get_semantic_impact_prediction(self, text, timeframe):
        """Predict price impact based on semantic similarity to historical patterns"""
        try:
            embedding = self.get_embedding(text)
            similar_news = self.find_similar_news(embedding)

            if not similar_news:
                return None

            weighted_impacts = []
            total_weight = 0

            for idx, similarity in similar_news:
                if timeframe in self.price_impacts[idx]:
                    impact = self.price_impacts[idx][timeframe]
                    weight = similarity
                    weighted_impacts.append(impact * weight)
                    total_weight += weight

            if total_weight == 0:
                return None

            return sum(weighted_impacts) / total_weight
        except Exception as e:
            logger.error(f"Error in semantic impact prediction: {str(e)}")
            return None

class NewsAggregator:
    def __init__(self):
        self.stock_predictions = {}  # Store predictions by stock and timeframe
        

    def add_prediction(self, symbol, timeframe, prediction, publish_time, deadline):
        if symbol not in self.stock_predictions:
            self.stock_predictions[symbol] = {}
        
        if timeframe not in self.stock_predictions[symbol]:
            self.stock_predictions[symbol][timeframe] = []
            
        self.stock_predictions[symbol][timeframe].append({
            'prediction': prediction,
            'publish_time': publish_time,
            'deadline': deadline
        })

    def get_aggregated_prediction(self, symbol, timeframe):
        if symbol in self.stock_predictions and timeframe in self.stock_predictions[symbol]:
            predictions = self.stock_predictions[symbol][timeframe]
            # Calculate weighted average based on recency
            total_weight = 0
            weighted_sum = 0
            current_time = datetime.now(tz=pytz.UTC)
            
            for pred in predictions:
                time_diff = (current_time - pred['publish_time']).total_seconds()
                weight = 1 / (time_diff + 1)  # More recent predictions get higher weight
                weighted_sum += pred['prediction'] * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0
        return None

class RealTimeMonitor:
    def __init__(self):
        logger.info("Initializing RealTimeMonitor")
        self.portfolio_tracker = PortfolioTrackerService()
        logger.info("Portfolio tracker service initialized")
        
        # Initialize model manager and load models
        self.model_manager = ModelManager()
        if not self.model_manager.load_models():
            logger.error("Failed to load models")
            raise RuntimeError("Model initialization failed")
            
        self.models = self.model_manager.models
        self.vectorizers = self.model_manager.vectorizers
        self.scalers = self.model_manager.scalers
        logger.info("Models loaded successfully")
        
        # Initialize prediction components
        self.market_trainer = MarketMLTrainer()
        
        # Add price cache with limits
        self.price_cache = {}
        self.price_cache_ttl = 60  # Cache prices for 60 seconds
        self.price_cache_max_size = 1000  # Maximum number of cached prices
        self.last_price_fetch = {}  # Track last fetch time for rate limiting
        self.price_fetch_delay = 2  # Minimum seconds between fetches for same symbol
        
        # Initialize Telegram settings
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_session = None
        
        # Initialize processed news with TTL
        self.processed_news = {}  # Change to dict to store timestamps
        self.processed_news_ttl = 86400  # Clear news older than 24 hours
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 3600  # Cleanup every hour
        
        # Share embedding model and analyzers from model manager
        self.embedding_model = self.model_manager.embedding_model
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.semantic_analyzer = self.model_manager.semantic_analyzer
        logger.info("Analyzers initialized")
        
        # Prediction thresholds
        self.prediction_thresholds = {
            '1wk': 10.0,  # 10% minimum for weekly predictions
            '1mo': 20.0   # 20% minimum for monthly predictions
        }
        
        # Time window for news
        self.max_news_age_hours = 24
        
        # Maximum allowed prior price increase
        self.max_prior_price_increase = 5.0  # 5%
        
        # CPU Management - Use all available CPUs
        self.max_workers = multiprocessing.cpu_count()
        logger.info(f"Using all {self.max_workers} available CPUs")
        
        # Memory management
        self.process = psutil.Process(os.getpid())
        self.memory_warning_threshold = 60.0
        self.memory_critical_threshold = 75.0
        
        # Processing settings
        self.batch_size = 100  # Increased batch size for more parallel processing
        self.delay_between_batches = 1  # Reduced delay between batches
        self.max_concurrent_symbols = self.max_workers * 2  # Double the number of concurrent symbols
        
        # Initialize other components
        self._polling_lock = asyncio.Lock()
        self._is_polling = False

    def _cleanup_caches(self):
        """Clean up old cache entries"""
        try:
            current_time = time.time()
            
            # Only cleanup if enough time has passed
            if current_time - self.last_cache_cleanup < self.cleanup_interval:
                return
                
            # Cleanup price cache
            expired_prices = []
            for symbol, (cache_time, _) in self.price_cache.items():
                if current_time - cache_time > self.price_cache_ttl:
                    expired_prices.append(symbol)
            for symbol in expired_prices:
                del self.price_cache[symbol]
            
            # Cleanup processed news
            expired_news = []
            for article_id, process_time in self.processed_news.items():
                if current_time - process_time > self.processed_news_ttl:
                    expired_news.append(article_id)
            for article_id in expired_news:
                del self.processed_news[article_id]
            
            # Enforce maximum cache sizes
            if len(self.price_cache) > self.price_cache_max_size:
                # Remove oldest entries
                sorted_cache = sorted(self.price_cache.items(), key=lambda x: x[1][0])
                self.price_cache = dict(sorted_cache[-self.price_cache_max_size:])
            
            self.last_cache_cleanup = current_time
            gc.collect()
            
            logger.info(f"Cache cleanup completed. Sizes - Prices: {len(self.price_cache)}, News: {len(self.processed_news)}")
            
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")

    async def _get_news_for_symbol(self, symbol):
        """Get news articles for a symbol with error handling"""
        try:
            # Use yf.Search instead of ticker.news
            search = yf.Search(
                query=symbol,
                news_count=8,
                include_nav_links=False,
                include_research=True
            )
            news = search.news.copy()  # Make a copy of the news data
            
            # Clear search object immediately
            del search
            gc.collect()
            
            return news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return []
        finally:
            gc.collect()

    async def _validate_article(self, article, current_time):
        """Validate article data and check if it should be processed"""
        try:
            if not isinstance(article, dict):
                logger.warning("Invalid article format")
                return None, None
                
            publish_time = article.get('providerPublishTime')
            if not publish_time:
                logger.warning(f"Missing publish time for article: {article.get('title', 'No title')}")
                return None, None
                
            article_link = article.get('link')
            if not article_link:
                logger.warning(f"Missing link for article: {article.get('title', 'No title')}")
                return None, None
            
            # Convert timestamp to datetime
            try:
                publish_date = pd.Timestamp(publish_time, unit='s', tz='UTC')
            except Exception as e:
                logger.error(f"Error converting publish time {publish_time}: {str(e)}")
                return None, None
                
            time_diff = current_time - publish_date
            
            # Check if article is recent enough
            if time_diff.total_seconds() > (24 * 3600):  # older than 24 hours
                return None, None
                
            # Create unique article ID
            article_id = article.get('uuid', '')
            if not article_id:
                title = article.get('title', '')
                article_id = f"{title}_{publish_time}"
                
            return article_id, publish_date
            
        except Exception as e:
            logger.error(f"Error validating article: {str(e)}")
            return None, None
        finally:
            gc.collect()

    async def monitor_stock(self, symbol):
        """Monitor a single stock for trading opportunities"""
        try:
            # Cleanup caches periodically
            self._cleanup_caches()
            
            # Get current price first for comparison
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return

            # Get news with error handling
            news = await self._get_news_for_symbol(symbol)
            if not news:
                return

            valid_predictions = []
            current_time = datetime.now(tz=timezone.utc)
            
            for article in news:
                try:
                    # Validate article and get publish date
                    article_id, publish_date = await self._validate_article(article, current_time)
                    if not article_id or not publish_date:
                        continue
                        
                    # Skip if already processed
                    if article_id in self.processed_news:
                        continue
                        
                    self.processed_news[article_id] = current_time
                    
                    # Get article content with timeout
                    try:
                        content = await asyncio.wait_for(
                            self.get_full_article_text(article['link']),
                            timeout=10
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout getting content for article: {article.get('title', 'No title')}")
                        continue
                    except Exception as e:
                        logger.error(f"Error getting content: {str(e)}")
                        continue
                    
                    if not content:
                        logger.warning(f"No content for article: {article.get('title', 'No title')}")
                        continue

                    logger.info(f"\nAnalyzing article from {publish_date}")
                    logger.info(f"Title: {article.get('title', 'No title')}")
                    logger.info(f"Publisher: {article.get('publisher', 'Unknown')}")
                    
                    # Make predictions for both timeframes
                    predictions = {}
                    for timeframe in ['1wk', '1mo']:
                        try:
                            prediction = await self._get_prediction(
                                symbol=symbol,
                                content=content,
                                timeframe=timeframe,
                                publish_time=publish_date,
                                current_price=current_price
                            )
                            
                            if prediction and prediction['prediction'] > 0:
                                predictions[timeframe] = prediction
                                logger.info(f"Valid {timeframe} prediction: {prediction['prediction']:.2f}%")
                                
                        except Exception as pred_error:
                            logger.error(f"Error getting prediction for {symbol} {timeframe}: {str(pred_error)}")
                            continue
                    
                    if predictions:
                        valid_predictions.append({
                            'article': article,
                            'predictions': predictions,
                            'publish_time': publish_date
                        })
                        logger.info(f"Added valid prediction for {symbol} based on news from {publish_date}")
                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                    continue

            # If we have valid predictions, analyze and notify
            if valid_predictions:
                await self.analyze_and_notify(symbol, valid_predictions, {'current_price': current_price})

            # Clear memory
            del valid_predictions
            gc.collect()

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")
        finally:
            # Ensure memory is cleaned up
            gc.collect()

    async def _get_prediction(self, symbol, content, timeframe, publish_time, current_price):
        """Get prediction for a specific timeframe with improved error handling"""
        try:
            logger.info("\n" + "="*50)
            logger.info(f"PREDICTION ANALYSIS: {symbol} - {timeframe}")
            logger.info("="*50)
            
            # Log the content being analyzed
            logger.info(f"Analyzing content (first 200 chars):\n{content[:200]}...\n")
            
            # 1. Get base ML prediction
            logger.info("1. ML Model Prediction:")
            try:
                # Get prediction using model manager
                ml_prediction = self.model_manager.predict(content, timeframe)
                if ml_prediction is not None:
                    weighted_ml = ml_prediction * 0.4
                    logger.info(f"   Raw ML prediction: {ml_prediction:.2f}%")
                    logger.info(f"   Weighted ML (40%): {weighted_ml:.2f}%")
                else:
                    logger.error("Model prediction returned None")
                    return None
            except Exception as e:
                logger.error(f"Error in ML prediction: {str(e)}")
                return None

            # 2. Get semantic prediction
            logger.info("\n2. Semantic Analysis:")
            try:
                semantic_pred = self.semantic_analyzer.get_semantic_impact_prediction(content, timeframe)
                if semantic_pred is not None:
                    weighted_semantic = semantic_pred * 0.4
                    logger.info(f"   Raw Semantic prediction: {semantic_pred:.2f}%")
                    logger.info(f"   Weighted Semantic (40%): {weighted_semantic:.2f}%")
                else:
                    weighted_semantic = None
                    logger.info("   No semantic prediction available")
            except Exception as e:
                logger.error(f"Error in semantic analysis: {str(e)}")
                weighted_semantic = None

            # 3. Get sentiment
            logger.info("\n3. Sentiment Analysis:")
            try:
                sentiment = self.finbert_analyzer.analyze_sentiment(content)
                if sentiment:
                    logger.info(f"   Score: {sentiment['score']:.2f}")
                    logger.info(f"   Confidence: {sentiment['confidence']:.2f}")
                    logger.info("   Probabilities:")
                    logger.info(f"      Positive: {sentiment['probabilities']['positive']:.2f}")
                    logger.info(f"      Negative: {sentiment['probabilities']['negative']:.2f}")
                    logger.info(f"      Neutral:  {sentiment['probabilities']['neutral']:.2f}")
                else:
                    logger.info("   Failed to get sentiment")
                    return None
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                return None

            # Calculate final prediction
            logger.info("\n4. Final Prediction Calculation:")
            try:
                if weighted_semantic is not None:
                    weighted_pred = (
                        weighted_ml + 
                        weighted_semantic + 
                        (weighted_ml * sentiment['score'] * 0.2)
                    )
                    logger.info("   Using ML + Semantic + Sentiment formula")
                else:
                    weighted_pred = weighted_ml * (1 + sentiment['score'] * 0.2)
                    logger.info("   Using ML + Sentiment formula")

                logger.info(f"   Final prediction: {weighted_pred:.2f}%")
                logger.info(f"   Required threshold: {self.prediction_thresholds[timeframe]:.2f}%")

                # Check minimum threshold
                if weighted_pred < self.prediction_thresholds[timeframe]:
                    logger.info("\nPrediction rejected: Below threshold")
                    logger.info("="*50)
                    return None

                logger.info("\nPrediction accepted!")
                logger.info(f"Time frame: {timeframe}")
                logger.info(f"Current price: ${current_price:.2f}")
                logger.info(f"Target price: ${current_price * (1 + weighted_pred/100):.2f}")
                logger.info("="*50)

                return {
                    'prediction': weighted_pred,
                    'confidence': sentiment['confidence'],
                    'sentiment_score': sentiment['score'],
                    'components': {
                        'ml_prediction': weighted_ml,
                        'semantic_prediction': weighted_semantic,
                        'sentiment_score': sentiment['score']
                    }
                }
            except Exception as e:
                logger.error(f"Error in final prediction calculation: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None
        finally:
            # Clean up any large objects
            gc.collect()

    async def analyze_and_notify(self, symbol, valid_predictions, price_data):
        """Analyze predictions and send notifications if needed"""
        try:
            current_price = price_data.get('current_price')
            if not current_price:
                logger.error(f"No current price available for {symbol}")
                return

            # Track best prediction for each timeframe
            best_predictions = {
                '1wk': {'prediction': None, 'article': None, 'score': 0},
                '1mo': {'prediction': None, 'article': None, 'score': 0}
            }

            # Analyze each prediction
            for pred_data in valid_predictions:
                article = pred_data['article']
                predictions = pred_data['predictions']
                publish_time = pred_data['publish_time']

                for timeframe, prediction in predictions.items():
                    # Calculate prediction score based on confidence and prediction value
                    score = prediction['prediction'] * prediction['confidence']
                    
                    # Update best prediction if this is better
                    if score > best_predictions[timeframe]['score']:
                        best_predictions[timeframe] = {
                            'prediction': prediction,
                            'article': article,
                            'score': score,
                            'publish_time': publish_time
                        }

            # Process signals for each timeframe
            for timeframe, best in best_predictions.items():
                if not best['prediction']:
                    continue

                prediction = best['prediction']
                article = best['article']
                publish_time = best['publish_time']

                # Calculate target price
                predicted_change = prediction['prediction'] / 100  # Convert to decimal
                target_price = current_price * (1 + predicted_change)

                # Set entry and target dates
                entry_date = datetime.now(tz=timezone.utc)
                target_date = entry_date + timedelta(days=7 if timeframe == '1wk' else 30)

                try:
                    # Send signal
                    await self.send_signal(
                        signal_type='buy',
                        symbol=symbol,
                        price=current_price,
                        target_price=target_price,
                        sentiment_score=prediction['sentiment_score'],
                        timeframe=timeframe,
                        reason=f"News-based prediction: {prediction['prediction']:.2f}% upside",
                        entry_date=entry_date,
                        target_date=target_date
                    )
                    
                    logger.info(f"Signal sent for {symbol} ({timeframe})")
                    logger.info(f"Predicted change: {predicted_change:.2%}")
                    logger.info(f"Target price: ${target_price:.2f}")
                except Exception as e:
                    logger.error(f"Error sending signal for {symbol} {timeframe}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in analyze_and_notify for {symbol}: {str(e)}")
        finally:
            # Clean up
            gc.collect()

    async def _get_historical_price(self, symbol, days=1):
        """Get historical price data with caching"""
        try:
            cache_key = f"{symbol}_{days}"
            current_time = time.time()
            
            # Check cache first
            if hasattr(self, 'historical_price_cache'):
                if cache_key in self.historical_price_cache:
                    cache_time, cached_data = self.historical_price_cache[cache_key]
                    # Cache for 1 hour for historical data
                    if current_time - cache_time < 3600:  # 1 hour in seconds
                        return cached_data
            else:
                self.historical_price_cache = {}
            
            # Rate limiting check
            if hasattr(self, 'last_historical_fetch'):
                if symbol in self.last_historical_fetch:
                    time_since_last_fetch = current_time - self.last_historical_fetch[symbol]
                    if time_since_last_fetch < self.price_fetch_delay:
                        await asyncio.sleep(self.price_fetch_delay - time_since_last_fetch)
            else:
                self.last_historical_fetch = {}
            
            # Update last fetch time
            self.last_historical_fetch[symbol] = current_time
            
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=f"{days}d")
            
            if history.empty:
                logger.error(f"No historical data available for {symbol}")
                return None
                
            # Calculate required values
            result = {
                'open': history['Open'].iloc[0],
                'close': history['Close'].iloc[-1],
                'high': history['High'].max(),
                'low': history['Low'].min(),
                'volume': history['Volume'].sum(),
                'dates': history.index.tolist()
            }
            
            # Cache the result
            self.historical_price_cache[cache_key] = (current_time, result)
            
            # Clear ticker to free memory
            del ticker
            gc.collect()
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting historical price for {symbol}: {str(e)}")
            return None
        finally:
            if 'ticker' in locals():
                del ticker
            if 'history' in locals():
                del history
            gc.collect()

    async def send_telegram_alert(self, message: str) -> None:
        """Send alert message via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram bot or chat ID not configured")
            return

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Create bot without session
                application = Application.builder().token(self.telegram_token).build()
                bot = application.bot
                
                await bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                logger.info("Telegram alert sent successfully")
                return
            except Exception as e:
                logger.error(f"Failed to send Telegram alert (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed message content:\n{message}")
                    break

    async def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            # Close portfolio tracker
            await self.portfolio_tracker.close()
            logger.info("Portfolio tracker session closed")
            
            # Close telegram session if exists
            if self.telegram_session:
                await self.telegram_session.close()
                logger.info("Telegram session closed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def start(self, symbols: list[str]):
        """Start the monitoring process with optimized batch processing"""
        try:
            startup_message = (
                "🚀 <b>Stock Monitor Activated</b>\n\n"
                f"📊 Monitoring {len(symbols)} stocks\n"
                f"🕒 Started at: {datetime.now(tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"💻 Using {self.max_workers} CPUs for processing\n"
                f"⚡ Processing {self.batch_size} symbols per batch\n"
                f"🔄 {self.max_concurrent_symbols} symbols processed concurrently"
            )
            await self.send_telegram_alert(startup_message)
            
            self._is_polling = True
            
            while True:
                try:
                    logger.info("Starting new monitoring cycle...")
                    self._cleanup_caches()  # Cleanup at the start of each cycle
                    
                    # Process in optimized batches
                    total_symbols = len(symbols)
                    for batch_start in range(0, total_symbols, self.batch_size):
                        batch = symbols[batch_start:batch_start + self.batch_size]
                        batch_number = batch_start // self.batch_size + 1
                        total_batches = (total_symbols + self.batch_size - 1) // self.batch_size
                        
                        logger.info(f"Processing Batch {batch_number}/{total_batches}")
                        
                        # Process symbols concurrently
                        sem = asyncio.Semaphore(self.max_concurrent_symbols)
                        tasks = []
                        
                        async def process_with_semaphore(symbol):
                            async with sem:
                                try:
                                    return await self.monitor_stock(symbol)
                                finally:
                                    gc.collect()  # Cleanup after each symbol
                        
                        for symbol in batch:
                            task = asyncio.create_task(process_with_semaphore(symbol))
                            tasks.append(task)
                            
                        await asyncio.gather(*tasks)
                        
                        # Clear completed tasks
                        tasks.clear()
                        gc.collect()
                        
                        logger.info(f"Completed Batch {batch_number}/{total_batches}")
                        
                        if batch_number < total_batches:
                            await asyncio.sleep(self.delay_between_batches)
                    
                    logger.info("Completed full monitoring cycle")
                    await asyncio.sleep(30)  # Wait 30 seconds before next cycle
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {str(e)}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Error in start: {str(e)}")
            self._is_polling = False
        finally:
            # Ensure cleanup happens
            await self.cleanup()

    async def test_portfolio_tracker(self):
        """Test function to verify portfolio tracker communication"""
        try:
            logger.info("=== Starting Portfolio Tracker Test ===")
            
            # Test data
            symbol = "AAPL"
            entry_price = 180.50
            target_price = 200.00
            entry_date = datetime.now(tz=pytz.UTC)
            target_date = entry_date + timedelta(days=30)
            
            logger.info(f"Sending test buy signal for {symbol}")
            
            # Send test buy signal
            success = await self.portfolio_tracker.send_buy_signal(
                        symbol=symbol,
                entry_price=entry_price,
                        target_price=target_price,
                entry_date=entry_date.isoformat(),
                target_date=target_date.isoformat()
            )
            
            if success:
                logger.info("✅ Portfolio tracker test successful")
            else:
                logger.error("❌ Portfolio tracker test failed")

        except Exception as e:
            logger.error(f"Portfolio tracker test failed: {str(e)}")
            raise  # Re-raise to prevent startup if test fails

    async def _get_current_price(self, symbol):
        """Get real-time current price for a symbol with caching and rate limiting"""
        try:
            current_time = time.time()
            
            # Check cache first (use shorter cache time for real-time data)
            if symbol in self.price_cache:
                cache_time, cached_price = self.price_cache[symbol]
                # Only cache for 30 seconds for real-time data
                if current_time - cache_time < 30:
                    return cached_price
            
            # Rate limiting check
            if symbol in self.last_price_fetch:
                time_since_last_fetch = current_time - self.last_price_fetch[symbol]
                if time_since_last_fetch < self.price_fetch_delay:
                    await asyncio.sleep(self.price_fetch_delay - time_since_last_fetch)
            
            # Update last fetch time
            self.last_price_fetch[symbol] = current_time
            
            # Get real-time price
            ticker = yf.Ticker(symbol)
            history = ticker.history(
                period="1d",          # Get today's data
                interval="1m",        # 1-minute intervals
                prepost=True,         # Include pre/post market
                repair=True,          # Repair any missing data
                keepna=False          # Remove any NA values
            )
            
            if not history.empty:
                # Get most recent price
                last_row = history.iloc[-1]
                price = last_row['Close']
                
                # Log price details for debugging
                logger.debug(f"Latest price for {symbol}:")
                logger.debug(f"Time: {history.index[-1]}")
                logger.debug(f"Open: {last_row['Open']:.2f}")
                logger.debug(f"High: {last_row['High']:.2f}")
                logger.debug(f"Low: {last_row['Low']:.2f}")
                logger.debug(f"Close: {last_row['Close']:.2f}")
                logger.debug(f"Volume: {last_row['Volume']}")
                
                # Update cache with shorter TTL for real-time data
                self.price_cache[symbol] = (current_time, float(price))
                return float(price)
            
            logger.error(f"No price data available for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
        finally:
            # Clean up
            if 'ticker' in locals():
                del ticker
            if 'history' in locals():
                del history
            gc.collect()

    def _format_recommendation_message(self, recommendation):
        """Format recommendation details for Telegram notification"""
        symbol = recommendation['symbol']
        message = f"🚨 *NEW TRADING SIGNAL*\n\n"
        message += f"Symbol: *{symbol}*\n"
        message += f"Combined Score: *{recommendation['score']:.2%}*\n\n"
        
        if recommendation['short_term']:
            st = recommendation['short_term']
            message += "*1-Week Analysis:*\n"
            message += f"• Predicted Change: {st['predicted_change']:.2%}\n"
            message += f"• Confidence: {st['confidence']:.2f}\n"
            message += f"• Signal Strength: {st['signal_strength']:.2%}\n"
            
            # Add sentiment and cluster info if available
            if 'sentiment_score' in st['analysis_components']:
                message += f"• Sentiment Score: {st['analysis_components']['sentiment_score']:.2f}\n"
            if 'cluster_impact' in st['analysis_components'] and st['analysis_components']['cluster_impact']:
                message += f"• Cluster Impact: {st['analysis_components']['cluster_impact']:.2%}\n"
        
        if recommendation['long_term']:
            lt = recommendation['long_term']
            message += "\n*1-Month Analysis:*\n"
            message += f"• Predicted Change: {lt['predicted_change']:.2%}\n"
            message += f"• Confidence: {lt['confidence']:.2f}\n"
            message += f"• Signal Strength: {lt['signal_strength']:.2%}\n"
        
        message += "\n💡 *Trading Notes:*\n"
        if recommendation['short_term'] and recommendation['long_term']:
            message += "• Signal confirmed in both timeframes (higher confidence)\n"
        message += f"• Stop Loss: {self.stop_loss_percentage:.1f}%\n"
        
        return message

    async def get_full_article_text(self, url):
        """Get full text content from news article URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text based on common article containers
            article_content = soup.find(['article', 'main', 'div'], 
                                     {'class': ['article', 'content', 'article-content']})
            if article_content:
                return article_content.get_text(separator=' ', strip=True)
        
            return None
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
            return None

def get_all_symbols():
    """Get list of stock symbols to monitor"""
    try:
        with open('stock_tickers.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols to monitor")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {str(e)}")
        return []

async def main():
    logger.info("🚀 Market Monitor Starting...")
    logger.info("Initializing ML models and market data...")
    monitor = RealTimeMonitor()
    
    try:
        # Run the portfolio tracker test first
        logger.info("Running portfolio tracker test...")
        await monitor.test_portfolio_tracker()
        
        # Then continue with normal operation
        symbols = get_all_symbols()
        await monitor.start(symbols)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        # Ensure cleanup happens even on error
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())




