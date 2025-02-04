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
import xgboost as xgb
import torch.nn as nn

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

class NewsLSTM(nn.Module):
    """LSTM model for news impact prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size  # Now required parameter
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Ensure input is 3D: [batch_size, seq_length, input_size]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        out = self.fc(last_output)
        
        return out, None  # Return None for attention as we've simplified the model

class HybridMarketPredictor(BaseEstimator, ClassifierMixin):
    """Model for predicting news impact on stock prices"""
    
    def __init__(self):
        self.n_features_in_ = None
        self.linear_weights = None
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.lstm_model = None  # Initialize as None, will be set after getting feature size
        
    def predict(self, X):
        """Make predictions using trained models"""
        try:
            if self.linear_weights is None:
                raise ValueError("Model not trained")
                
            # Convert sparse matrix to dense if needed
            if scipy.sparse.issparse(X):
                X = X.toarray()
            
            # Initialize LSTM if not done yet
            if self.lstm_model is None and hasattr(self, 'n_features_in_'):
                self.lstm_model = NewsLSTM(input_size=self.n_features_in_)
            
            # Get predictions from each model
            predictions = []
            
            # 1. Linear Model
            try:
                X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                linear_pred = X_with_bias @ self.linear_weights
                predictions.append(linear_pred)
            except Exception as e:
                logger.error(f"Error in linear prediction: {str(e)}")
            
            # 2. XGBoost
            try:
                xgb_pred = self.xgb_model.predict(X)
                predictions.append(xgb_pred)
            except Exception as e:
                logger.error(f"Error in XGBoost prediction: {str(e)}")
            
            # 3. LSTM
            try:
                if self.lstm_model is not None:
                    X_tensor = torch.FloatTensor(X)
                    self.lstm_model.eval()
                    with torch.no_grad():
                        lstm_pred, _ = self.lstm_model(X_tensor)
                        predictions.append(lstm_pred.numpy().squeeze())
            except Exception as e:
                logger.error(f"Error in LSTM prediction: {str(e)}")
            
            # Combine predictions
            if predictions:
                final_predictions = np.mean(predictions, axis=0)
                return np.clip(final_predictions, -50, 50)  # Clip to reasonable range
            else:
                raise ValueError("All models failed to make predictions")
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

class ModelManager:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_analyzer = NewsSemanticAnalyzer(embedding_model=self.embedding_model)
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.model_paths = {
            '1wk': {
                'model': 'app/models/market_model_1wk.joblib',
                'vectorizer': 'app/models/vectorizer_1wk.joblib'
            },
            '1mo': {
                'model': 'app/models/market_model_1mo.joblib',
                'vectorizer': 'app/models/vectorizer_1mo.joblib'
            }
        }

    def predict(self, text, timeframe):
        """Make a prediction using the loaded models"""
        try:
            if timeframe not in self.models:
                logger.error(f"No model loaded for timeframe: {timeframe}")
                return None
                
            # Get model components
            model = self.models[timeframe]
            vectorizer = self.vectorizers[timeframe]
            
            # Transform text using vectorizer
            X_tfidf = vectorizer.transform([text])
            
            # Get sentiment features
            sentiment = self.finbert_analyzer.analyze_sentiment(text)
            if not sentiment:
                logger.error("Failed to analyze sentiment")
                return None
            
            # Get semantic embedding
            embedding = self.semantic_analyzer.get_embedding(text)
            if embedding is None:
                logger.error("Failed to get semantic embedding")
                return None
            
            # Make prediction
            prediction = model.predict(X_tfidf)
            
            # Return prediction with details
            return {
                'timeframe': timeframe,
                'price_impact': float(prediction[0]),
                'sentiment': sentiment,
                'confidence': sentiment['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

    def load_models(self):
        """Load all models and components"""
        try:
            logger.info("Loading models and components...")
            
            # Load models for each timeframe
            for timeframe in ['1wk', '1mo']:
                paths = self.model_paths[timeframe]
                
                # Check if files exist
                if not os.path.exists(paths['model']) or not os.path.exists(paths['vectorizer']):
                    logger.error(f"Model files not found for {timeframe}")
                    return False
                
                # Load components
                self.models[timeframe] = joblib.load(paths['model'])
                self.vectorizers[timeframe] = joblib.load(paths['vectorizer'])
                
                logger.info(f"Loaded model components for {timeframe}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def check_and_reload_models(self):
        """Check if models need to be reloaded"""
        try:
            current_time = datetime.now()
            should_reload = False

            # Check model files
            for timeframe in ['1wk', '1mo']:
                paths = self.model_paths[timeframe]
                for path in paths.values():
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

class NewsSemanticAnalyzer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # Don't load additional BERT models if we have embedding_model
        if not hasattr(self.embedding_model, 'encode'):
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.tokenizer = None
            self.model = None

        # Use disk storage for embeddings
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.embeddings_dir = os.path.join(self.data_dir, 'embeddings')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Keep minimal in-memory data
        self.embedding_cache = {}
        self.cache_size_limit = 100  # Only keep last 100 embeddings in memory
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Cleanup every 5 minutes

    def cleanup_memory(self):
        """Periodic memory cleanup"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        try:
            # Clear embedding cache if too large
            if len(self.embedding_cache) > self.cache_size_limit:
                # Keep only most recent entries
                sorted_cache = sorted(self.embedding_cache.items(), key=lambda x: x[1]['timestamp'])
                self.embedding_cache = dict(sorted_cache[-self.cache_size_limit:])

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.last_cleanup = current_time

        except Exception as e:
            logger.error(f"Error in memory cleanup: {str(e)}")

    def get_embedding(self, text):
        """Generate embedding for text using existing embedding model or BERT"""
        try:
            self.cleanup_memory()
            
            # Check cache first
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]['embedding']

            # Generate embedding
            if hasattr(self.embedding_model, 'encode'):
                embedding = self.embedding_model.encode(text)
            else:
                inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embedding = embedding[0].numpy()

            # Update cache
            self.embedding_cache[text_hash] = {
                'embedding': embedding,
                'timestamp': time.time()
            }

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
        finally:
            # Cleanup references
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'token_embeddings' in locals():
                del token_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def find_similar_news(self, embedding, threshold=0.8):
        """Find similar historical news articles based on embedding similarity"""
        try:
            if not self.embedding_cache:
                return []
            similarities = cosine_similarity([embedding], list(self.embedding_cache.values()))[0]
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
                if timeframe in self.embedding_cache[idx]['embedding']:
                    impact = self.embedding_cache[idx]['embedding'][timeframe]
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
        
        # Share components from model manager
        self.embedding_model = self.model_manager.embedding_model
        self.finbert_analyzer = self.model_manager.finbert_analyzer
        self.semantic_analyzer = self.model_manager.semantic_analyzer
        logger.info("Models and analyzers initialized")
        
        # Use LRU cache for prices with size limit
        self.price_cache = {}
        self.price_cache_ttl = 60  # Cache prices for 60 seconds
        self.price_cache_max_size = 100  # Reduced from 1000
        self.last_price_fetch = {}
        self.price_fetch_delay = 2
        
        # Initialize Telegram settings
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_session = None
        
        # Initialize processed news with TTL
        self.processed_news = {}
        self.processed_news_ttl = 3600  # Reduced from 86400 to 1 hour
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 300  # Reduced from 3600 to 5 minutes
        
        # Prediction thresholds
        self.prediction_thresholds = {
            '1wk': 10.0,
            '1mo': 20.0
        }
        
        # Processing settings
        self.batch_size = 50  # Reduced from 100
        self.delay_between_batches = 2  # Increased from 1
        self.max_workers = min(multiprocessing.cpu_count(), 4)  # Limit max workers
        self.max_concurrent_symbols = self.max_workers  # Reduced from 2x
        
        # Memory management
        self.process = psutil.Process(os.getpid())
        self.memory_warning_threshold = 60.0
        self.memory_critical_threshold = 75.0
        
        # Initialize other components
        self._polling_lock = asyncio.Lock()
        self._is_polling = False
        
        logger.info(f"Initialized with {self.max_workers} workers and {self.batch_size} batch size")

    def _cleanup_caches(self):
        """Clean up memory and caches"""
        try:
            current_time = time.time()
            if current_time - self.last_cache_cleanup < self.cleanup_interval:
                return

            # Check memory usage
            memory_percent = self.process.memory_percent()
            if memory_percent > self.memory_warning_threshold:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                
                # Aggressive cleanup if critical
                if memory_percent > self.memory_critical_threshold:
                    self.price_cache.clear()
                    self.processed_news.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning("Performed emergency memory cleanup")
                    return

            # Normal cleanup
            # Clean price cache
            current_cache_size = len(self.price_cache)
            if current_cache_size > self.price_cache_max_size:
                # Remove oldest entries
                sorted_cache = sorted(self.price_cache.items(), key=lambda x: x[1][0])
                self.price_cache = dict(sorted_cache[-self.price_cache_max_size:])
                logger.info(f"Cleaned price cache: {current_cache_size} -> {len(self.price_cache)}")

            # Clean processed news
            old_news_count = len(self.processed_news)
            self.processed_news = {
                k: v for k, v in self.processed_news.items()
                if current_time - v < self.processed_news_ttl
            }
            if old_news_count > len(self.processed_news):
                logger.info(f"Cleaned processed news: {old_news_count} -> {len(self.processed_news)}")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.last_cache_cleanup = current_time
            
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")

    async def _get_news_for_symbol(self, symbol):
        """Get news articles with memory-efficient processing"""
        try:
            # Use yf.Ticker instead of Search for more focused results
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:
                # Only keep essential fields to reduce memory
                filtered_news = []
                for article in news:
                    filtered_article = {
                        'title': article.get('title', ''),
                        'link': article.get('link', ''),
                        'publisher': article.get('publisher', ''),
                        'providerPublishTime': article.get('providerPublishTime', 0),
                        'uuid': article.get('uuid', '')
                    }
                    filtered_news.append(filtered_article)
                news = filtered_news
            
            # Clear ticker object
            del ticker
            gc.collect()
            
            return news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return []
        finally:
            gc.collect()

    async def monitor_stock(self, symbol):
        """Monitor a single stock for trading opportunities"""
        try:
            # Get news articles
            news = await self._get_news_for_symbol(symbol)
            if not news:
                return
            
            current_time = pd.Timestamp.now(tz='UTC')
            predictions = []
            
            for article in news:
                try:
                    # Basic validation
                    if not isinstance(article, dict):
                        continue
                        
                    publish_time = article.get('providerPublishTime')
                    article_link = article.get('link')
                    if not publish_time or not article_link:
                        continue
                    
                    # Check if article is recent and not processed
                    publish_date = pd.Timestamp(publish_time, unit='s', tz='UTC')
                    if (current_time - publish_date).total_seconds() > self.processed_news_ttl:
                        continue
                        
                    article_id = article.get('uuid') or f"{article.get('title', '')}_{publish_time}"
                    if article_id in self.processed_news:
                        continue
                    
                    # Process article
                    content = f"{article.get('title', '')} {article.get('publisher', '')}"
                    
                    # Get predictions for each timeframe
                    for timeframe in self.prediction_thresholds:
                        prediction = await self._get_prediction(symbol, content, timeframe)
                        if prediction and prediction >= self.prediction_thresholds[timeframe]:
                            predictions.append({
                                'timeframe': timeframe,
                                'prediction': prediction,
                                'article': article
                            })
                    
                    # Mark article as processed
                    self.processed_news[article_id] = time.time()
                    
                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                finally:
                    # Clear any temporary objects
                    gc.collect()
            
            # Process valid predictions
            if predictions:
                await self._process_predictions(symbol, predictions)
            
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")
        finally:
            # Cleanup after processing
            self._cleanup_caches()
            gc.collect()

    async def _process_predictions(self, symbol, predictions):
        """Process valid predictions and send notifications"""
        try:
            for pred in predictions:
                timeframe = pred['timeframe']
                prediction = pred['prediction']
                article = pred['article']
                
                message = (
                    f"🚨 Trading Signal for {symbol}\n"
                    f"Timeframe: {timeframe}\n"
                    f"Prediction: {prediction:.2f}%\n"
                    f"Article: {article.get('title')}\n"
                    f"Source: {article.get('publisher')}\n"
                    f"Link: {article.get('link')}"
                )
                
                # Send notification
                if self.telegram_token and self.telegram_chat_id:
                    await self._send_telegram_message(message)
                
                logger.info(f"Trading signal generated for {symbol}: {prediction:.2f}% ({timeframe})")
                
        except Exception as e:
            logger.error(f"Error processing predictions for {symbol}: {str(e)}")
        finally:
            gc.collect()

    async def _get_prediction(self, symbol, content, timeframe):
        """Get prediction for a specific timeframe with improved error handling"""
        try:
            # 1. Get ML prediction from model manager
            ml_prediction = self.model_manager.predict(content, timeframe)
            if not ml_prediction:
                return None
                
            weighted_ml = ml_prediction['price_impact'] * 0.4
            
            # Early exit if ML prediction is too low
            if weighted_ml < (self.prediction_thresholds[timeframe] * 0.2):  # If ML component can't reach 20% of threshold
                return None
            
            # 2. Get sentiment (faster than semantic)
            sentiment = self.finbert_analyzer.analyze_sentiment(content)
            if not sentiment:
                return None
                
            # Early exit if sentiment is very negative
            if sentiment['score'] < -0.5:
                return None
            
            # 3. Get semantic prediction only if needed
            weighted_semantic = None
            if weighted_ml >= (self.prediction_thresholds[timeframe] * 0.3):  # Only get semantic if ML shows promise
                semantic_pred = self.semantic_analyzer.get_semantic_impact_prediction(content, timeframe)
                if semantic_pred is not None:
                    weighted_semantic = semantic_pred * 0.4
            
            # Calculate final prediction
            if weighted_semantic is not None:
                weighted_pred = (
                    weighted_ml + 
                    weighted_semantic + 
                    (weighted_ml * sentiment['score'] * 0.2)
                )
            else:
                weighted_pred = weighted_ml * (1 + sentiment['score'] * 0.2)
            
            # Log prediction details
            if weighted_pred >= self.prediction_thresholds[timeframe]:
                logger.info(f"\nValid prediction for {symbol} ({timeframe}):")
                logger.info(f"ML Impact: {weighted_ml:.2f}%")
                if weighted_semantic is not None:
                    logger.info(f"Semantic Impact: {weighted_semantic:.2f}%")
                logger.info(f"Sentiment Score: {sentiment['score']:.2f}")
                logger.info(f"Final Prediction: {weighted_pred:.2f}%")
            
            return weighted_pred if weighted_pred >= self.prediction_thresholds[timeframe] else None
            
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {str(e)}")
            return None
        finally:
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




