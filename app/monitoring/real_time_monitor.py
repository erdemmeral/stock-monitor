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

class ModelManager:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.semantic_patterns = None
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
            
            # Combine features
            X_combined = scipy.sparse.hstack([
                X_tfidf,
                scipy.sparse.csr_matrix(sentiment_features)
            ]).tocsr()
            
            # Get embedding for LSTM
            semantic_analyzer = NewsSemanticAnalyzer()
            embedding = semantic_analyzer.get_embedding(text)
            embeddings_sequence = [embedding] if embedding is not None else None
            
            # Make prediction
            prediction, details = model.predict(X_combined, embeddings_sequence)
            
            # Inverse transform the prediction
            prediction_unscaled = scaler.inverse_transform([[prediction]])[0][0]
            
            logger.info(f"\nPrediction for {timeframe}:")
            logger.info(f"  Raw prediction: {prediction:.4f}")
            logger.info(f"  Unscaled prediction: {prediction_unscaled:.2f}%")
            if details['lstm_pred'] is not None:
                logger.info(f"  XGBoost prediction: {details['xgb_pred']:.4f}")
                logger.info(f"  LSTM prediction: {details['lstm_pred']:.4f}")
            
            return prediction_unscaled
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.exception("Full traceback:")
            return None

class NewsSemanticAnalyzer:
    def __init__(self):
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

        # Initialize prediction components
        self.market_trainer = MarketMLTrainer()
        # Train models if they don't exist
        if not self.market_trainer.load_models():
            logger.info("No pre-trained models found. Training new models...")
            if not self.market_trainer.train():
                raise RuntimeError("Failed to train models")
        logger.info("Market ML models loaded")
        
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.semantic_analyzer = NewsSemanticAnalyzer()
        logger.info("Analyzers initialized")

        # Use models from market trainer
        self.vectorizers = self.market_trainer.vectorizers
        self.models = self.market_trainer.models
        
        logger.info("Models loaded successfully")
        
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
        self.processed_news = set()
        self._polling_lock = asyncio.Lock()
        self._is_polling = False

    async def _get_recent_news(self, symbol):
        """Get only very recent news for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return None
                
            # Get current time in UTC
            current_time = datetime.now(tz=timezone.utc)
            recent_news = []
            
            for article in news:
                # Convert timestamp to datetime
                publish_time = datetime.fromtimestamp(article['providerPublishTime'], tz=timezone.utc)
                time_diff = current_time - publish_time
                
                # Only include news from last 24 hours
                if time_diff.total_seconds() <= (24 * 3600):  # 24 hours in seconds
                    # Check if we've already processed this article
                    article_id = article.get('uuid', '')
                    if article_id not in self.processed_news:
                        self.processed_news.add(article_id)
                        recent_news.append(article)
                        logger.info(f"Found recent news for {symbol} published at {publish_time}")
                        logger.info(f"News title: {article.get('title', 'No title')}")
            
            if recent_news:
                logger.info(f"Found {len(recent_news)} recent news articles for {symbol}")
            
            return recent_news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return None

    async def monitor_stock(self, symbol):
        """Monitor a single stock for trading opportunities"""
        try:
            # Get current price first for comparison
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return

            # Get recent news articles
            news_items = await self._get_recent_news(symbol)
            if not news_items:
                return

            valid_predictions = []
            
            for article in news_items:
                try:
                    # Get article content
                    content = await self.get_full_article_text(article['link'])
                    if not content:
                        logger.warning(f"Could not get content for article: {article.get('title', 'No title')}")
                        continue

                    publish_time = datetime.fromtimestamp(article['providerPublishTime'], tz=timezone.utc)
                    logger.info(f"\nAnalyzing article from {publish_time}")
                    logger.info(f"Title: {article.get('title', 'No title')}")
                    
                    # Get predictions for both timeframes
                    predictions = {}
                    for timeframe in ['1wk', '1mo']:
                        prediction = await self._get_prediction(
                            symbol=symbol,
                            content=content,
                            timeframe=timeframe,
                            publish_time=publish_time,
                            current_price=current_price
                        )
                        
                        if prediction and prediction['prediction'] > 0:  # Only consider positive predictions
                            predictions[timeframe] = prediction
                            logger.info(f"Valid {timeframe} prediction: {prediction['prediction']:.2f}%")
                    
                    if predictions:  # If we have any valid predictions
                        valid_predictions.append({
                            'article': article,
                            'predictions': predictions,
                            'publish_time': publish_time
                        })
                        logger.info(f"Added valid prediction for {symbol} based on news from {publish_time}")

                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                    continue

            # If we have valid predictions, analyze and notify
            if valid_predictions:
                await self.analyze_and_notify(symbol, valid_predictions, {'current_price': current_price})

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")

    async def _get_prediction(self, symbol, content, timeframe, publish_time, current_price):
        """Get prediction for a specific timeframe"""
        try:
            logger.info("\n" + "="*50)
            logger.info(f"PREDICTION ANALYSIS: {symbol} - {timeframe}")
            logger.info("="*50)
            
            # Log the content being analyzed
            logger.info(f"Analyzing content (first 200 chars):\n{content[:200]}...\n")
            
            # 1. Get base ML prediction
            logger.info("1. ML Model Prediction:")
            X_tfidf = self.vectorizers[timeframe].transform([content])
            ml_prediction = self.models[timeframe].predict(X_tfidf)[0]
            weighted_ml = ml_prediction * 0.4
            logger.info(f"   Raw ML prediction: {ml_prediction:.2f}%")
            logger.info(f"   Weighted ML (40%): {weighted_ml:.2f}%")

            # 2. Get semantic prediction
            logger.info("\n2. Semantic Analysis:")
            semantic_pred = self.semantic_analyzer.get_semantic_impact_prediction(content, timeframe)
            if semantic_pred is not None:
                weighted_semantic = semantic_pred * 0.4
                logger.info(f"   Raw Semantic prediction: {semantic_pred:.2f}%")
                logger.info(f"   Weighted Semantic (40%): {weighted_semantic:.2f}%")
            else:
                weighted_semantic = None
                logger.info("   No semantic prediction available")

            # 3. Get sentiment
            logger.info("\n3. Sentiment Analysis:")
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

            # Calculate final prediction
            logger.info("\n4. Final Prediction Calculation:")
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
            logger.error(f"Error in prediction: {str(e)}")
            return None

    async def analyze_and_notify(self, symbol, valid_predictions, price_data):
        """Analyze predictions and send notifications if criteria met"""
        try:
            best_prediction = None
            highest_confidence = 0

            for pred_data in valid_predictions:
                # Check if either timeframe shows strong positive predictions
                week_pred = pred_data['predictions'].get('1wk', {}).get('prediction', 0)
                month_pred = pred_data['predictions'].get('1mo', {}).get('prediction', 0)

                # Signal is valid if either timeframe meets its threshold
                if (week_pred >= self.prediction_thresholds['1wk'] or 
                    month_pred >= self.prediction_thresholds['1mo']):
                    
                    # Calculate combined confidence
                    confidences = []
                    if week_pred >= self.prediction_thresholds['1wk']:
                        confidences.append(pred_data['predictions']['1wk']['confidence'])
                    if month_pred >= self.prediction_thresholds['1mo']:
                        confidences.append(pred_data['predictions']['1mo']['confidence'])
                    
                    avg_confidence = sum(confidences) / len(confidences)

                    if avg_confidence > highest_confidence:
                        highest_confidence = avg_confidence
                        best_prediction = pred_data

            if best_prediction:
                # Calculate target prices
                current_price = price_data['current_price']
                
                # Determine which timeframe to use based on strongest prediction
                week_pred = best_prediction['predictions'].get('1wk', {}).get('prediction', 0)
                month_pred = best_prediction['predictions'].get('1mo', {}).get('prediction', 0)
                
                # Use the timeframe with the stronger signal
                if (week_pred >= self.prediction_thresholds['1wk'] and 
                    (week_pred > month_pred or month_pred < self.prediction_thresholds['1mo'])):
                    # Weekly prediction is stronger or monthly doesn't meet threshold
                    timeframe = '1wk'
                    target_price = current_price * (1 + week_pred/100)
                    extended_target = current_price * (1 + month_pred/100) if month_pred > 0 else None
                    target_days = 7
                else:
                    # Monthly prediction is stronger
                    timeframe = '1mo'
                    target_price = current_price * (1 + month_pred/100)
                    extended_target = None
                    target_days = 30

                # Send signal
                await self.send_signal(
                    signal_type='buy',
                    symbol=symbol,
                    price=current_price,
                    target_price=target_price,
                    extended_target=extended_target,
                    sentiment_score=best_prediction['predictions'][timeframe]['sentiment_score'],
                    timeframe=timeframe,
                    reason=(
                        f"Strong {timeframe} signal: {best_prediction['predictions'][timeframe]['prediction']:.1f}% "
                        f"upside potential"
                    ),
                    entry_date=best_prediction['publish_time'],
                    target_date=best_prediction['publish_time'] + timedelta(days=target_days)
                )

        except Exception as e:
            logger.error(f"Error in analyze_and_notify for {symbol}: {str(e)}")

    async def _get_historical_price(self, symbol, days=1):
        """Get historical price for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if not hist.empty:
                return hist['Close'].iloc[0]
            return None
        except Exception as e:
            logger.error(f"Error getting historical price for {symbol}: {str(e)}")
            return None

    async def start(self, symbols: list[str]):
        """Start the monitoring process with optimized batch processing"""
        try:
            startup_message = (
                "🚀 <b>Stock Monitor Activated</b>\n\n"
                f"📊 Monitoring {len(symbols)} stocks\n"
                f"🕒 Started at: {datetime.now(tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"💻 Using {self.max_workers} CPUs for processing\n"
                f"⚡ Processing {self.batch_size} symbols per batch\n"
                f"🔄 {self.max_concurrent_symbols} symbols processed concurrently\n"
                "🔍 Initializing market monitoring process..."
            )
            await self.send_telegram_alert(startup_message)
            
            logger.info(f"Starting monitoring for {len(symbols)} symbols...")
            logger.info(f"Using {self.max_workers} CPUs for processing")
            
            self._is_polling = True
            polling_task = asyncio.create_task(self.poll_telegram_updates())
            model_check_task = asyncio.create_task(self.periodic_model_check())
            
            while True:
                try:
                    logger.info("Starting new monitoring cycle...")
                    
                    # Process in optimized batches
                    total_symbols = len(symbols)
                    for batch_start in range(0, total_symbols, self.batch_size):
                        batch = symbols[batch_start:batch_start + self.batch_size]
                        batch_number = batch_start // self.batch_size + 1
                        total_batches = (total_symbols + self.batch_size - 1) // self.batch_size
                        
                        logger.info(f"Processing Batch {batch_number}/{total_batches}")
                        
                        # Process symbols concurrently with increased parallelism
                        sem = asyncio.Semaphore(self.max_concurrent_symbols)
                        tasks = []
                        
                        async def process_with_semaphore(symbol):
                            async with sem:
                                return await self.monitor_stock(symbol)
                        
                        # Create tasks with increased concurrency
                        for symbol in batch:
                            task = asyncio.create_task(process_with_semaphore(symbol))
                            tasks.append(task)
                        
                        # Wait for all tasks in batch to complete
                        await asyncio.gather(*tasks)
                        
                        logger.info(f"Completed Batch {batch_number}/{total_batches}")
                        
                        # Add small delay between batches
                        if batch_number < total_batches:
                            await asyncio.sleep(self.delay_between_batches)
                    
                    logger.info("Completed full monitoring cycle")
                    await asyncio.sleep(30)  # Reduced wait time between cycles to 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {str(e)}")
                    await asyncio.sleep(30)
                    
        finally:
            self._is_polling = False
            self.thread_pool.shutdown()
            await polling_task
            await model_check_task

    def _adjust_features(self, X, expected_features):
        """Adjust feature matrix to match expected dimensions"""
        current_features = X.shape[1]
        logger.info(f"Current features: {current_features}, Expected: {expected_features}")
        
        if current_features < expected_features:
            # Pad with zeros
            padding = scipy.sparse.csr_matrix((X.shape[0], expected_features - current_features))
            X_adjusted = scipy.sparse.hstack([X, padding])
            logger.info(f"Padded features to {X_adjusted.shape[1]}")
            return X_adjusted
        elif current_features > expected_features:
            # Truncate
            X_adjusted = X[:, :expected_features]
            logger.info(f"Truncated features to {X_adjusted.shape[1]}")
            return X_adjusted
        return X

    def predict_with_sentiment(self, text, timeframe, symbol=None, publish_time=None):
        """
        Make a prediction integrating sentiment scores and semantic analysis
        """
        try:
            # 1. Get base ML prediction
            X_tfidf = self.vectorizers[timeframe].transform([text])
            
            # Log feature statistics for debugging
            logger.info(f"\nFeature Statistics for {timeframe}:")
            logger.info(f"Non-zero features: {X_tfidf.nnz}")
            logger.info(f"Max feature value: {X_tfidf.max()}")
            logger.info(f"Mean of non-zero values: {X_tfidf.sum() / X_tfidf.nnz if X_tfidf.nnz > 0 else 0}")
            
            # Check if feature dimensions match
            expected_features = self.models[timeframe].n_features_in_
            actual_features = X_tfidf.shape[1]
            
            if actual_features != expected_features:
                logger.info(f"Adjusting features from {actual_features} to {expected_features}")
                X_tfidf = self._adjust_features(X_tfidf, expected_features)
                if X_tfidf.shape[1] != expected_features:
                    logger.error(f"Feature adjustment failed. Got {X_tfidf.shape[1]}, expected {expected_features}")
                    return 0.0
            
            # Get raw prediction
            raw_prediction = self.models[timeframe].predict(X_tfidf)[0]
            logger.info(f"Raw prediction (before any processing): {raw_prediction}")
            
            # Log model coefficient stats
            logger.info("Model Coefficient Stats:")
            logger.info(f"Max coefficient: {np.max(self.models[timeframe].coef_)}")
            logger.info(f"Min coefficient: {np.min(self.models[timeframe].coef_)}")
            logger.info(f"Mean coefficient: {np.mean(self.models[timeframe].coef_)}")
            
            # Clip prediction to reasonable bounds based on timeframe
            max_change = self.min_price_changes[timeframe] * 10  # Allow up to 10x the threshold
            base_prediction = np.clip(raw_prediction, -max_change, max_change)
            logger.info(f"Clipped prediction: {base_prediction}")

            # Weight the predictions (40% ML, 40% semantic, 20% sentiment)
            ml_weight = 0.4
            semantic_weight = 0.4
            sentiment_weight = 0.2

            # 1. Base ML prediction (40%)
            weighted_ml = base_prediction * ml_weight
            logger.info(f"1. Base ML Prediction (40%): {base_prediction}% -> {weighted_ml}%")
            
            # 2. Get sentiment analysis
            sentiment = self.finbert_analyzer.analyze_sentiment(text)
            if not sentiment:
                logger.warning("No sentiment available for prediction")
                sentiment_multiplier = 1.0
            else:
                # Calculate sentiment multiplier
                score = sentiment['score']
                confidence = sentiment['confidence']
                neutral_prob = sentiment['probabilities']['neutral']

                # Calculate sentiment impact
                base_multiplier = 1.0 + (score * 0.3)  # Maps [-1, 1] to [0.7, 1.3]
                neutral_dampener = 1.0 - neutral_prob  # Reduce impact if sentiment is neutral
                sentiment_multiplier = 1.0 + (base_multiplier - 1.0) * confidence * neutral_dampener

            # 3. Get semantic prediction
            semantic_pred = self.semantic_analyzer.get_semantic_impact_prediction(text, timeframe)

            # 4. Combine predictions with weights
            if semantic_pred is not None:
                weighted_pred = (
                    0.4 * weighted_ml +  # 40% ML prediction
                    0.4 * semantic_pred +  # 40% semantic prediction
                    0.2 * (weighted_ml * sentiment_multiplier)  # 20% sentiment-adjusted prediction
                )
            else:
                weighted_pred = weighted_ml * sentiment_multiplier

            # Log prediction breakdown
            logger.info(f"\nPrediction Breakdown for {timeframe}:")
            logger.info(f"1. Base ML Prediction (40%): {weighted_ml:.2f}%")
            if semantic_pred is not None:
                logger.info(f"2. Semantic Prediction (40%): {semantic_pred:.2f}%")
            if sentiment:
                logger.info(f"3. Sentiment Score: {sentiment['score']:.2f}")
                logger.info(f"   Sentiment Multiplier: {sentiment_multiplier:.2f}")
            logger.info(f"Final Prediction: {weighted_pred:.2f}%")

            # Store prediction if symbol and publish_time provided
            if symbol and publish_time:
                if timeframe == '1h':
                    deadline = publish_time + timedelta(hours=1)
                elif timeframe == '1wk':
                    deadline = publish_time + timedelta(days=7)
                else:
                    deadline = publish_time + timedelta(days=30)
                
                self.news_aggregator.add_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    prediction=weighted_pred,
                    publish_time=publish_time,
                    deadline=deadline
                )
                
                # Send alert if prediction is significant
                if abs(weighted_pred) >= self.min_price_changes[timeframe]:
                    asyncio.create_task(self._send_prediction_alert(
                        symbol=symbol,
                        timeframe=timeframe,
                        prediction=weighted_pred,
                        news_text=text
                    ))

            return weighted_pred

        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return 0.0

    async def _send_prediction_alert(self, symbol: str, timeframe: str, prediction: float, news_text: str):
        """Internal method to send prediction alerts"""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                return
                
            timeframe_text = {
                '1h': '1 hour',
                '1wk': '1 week',
                '1mo': '1 month'
            }
            
            direction = "UP 📈" if prediction > 0 else "DOWN 📉"
            message = (
                f"🚨 Significant Movement Predicted!\n\n"
                f"Symbol: {symbol}\n"
                f"Timeframe: {timeframe_text[timeframe]}\n"
                f"Prediction: {direction} {abs(prediction):.2f}%\n\n"
                f"News Summary:\n{news_text[:200]}..."
            )
            
            bot = telegram.Bot(token=self.telegram_token)
            await bot.send_message(chat_id=self.telegram_chat_id, text=message)
            
        except Exception as e:
            logger.error(f"Error sending prediction alert: {str(e)}")

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
                text = possible_content.get_text(separator=' ', strip=True)
                logger.info(f"Successfully retrieved article text: {len(text)} characters")
                #logger.info(f"Article content: {text}")
                return text
                
            return None
        except Exception as e:
            logger.error(f"Error fetching article content: {e}")
            return None

    async def send_signal(self, signal_type: str, symbol: str, price: float, target_price: float, 
                         sentiment_score: float, timeframe: str, reason: str,
                         entry_date: Optional[datetime] = None, target_date: Optional[datetime] = None) -> bool:
            """Unified method to send buy/sell signals"""
            try:
                # Send signal to portfolio tracker
                if signal_type.lower() == 'buy':
                    if not entry_date:
                        entry_date = datetime.now(tz=pytz.UTC)
                    if not target_date:
                        target_date = entry_date + timedelta(days=30)
                    
                    # Format dates to match the expected format
                    entry_date_str = entry_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
                    target_date_str = target_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
                    
                    success = await self.portfolio_tracker.send_buy_signal(
                        symbol=symbol,
                        entry_price=float(price),
                        target_price=float(target_price),
                        entry_date=entry_date_str,
                        target_date=target_date_str
                    )
                else:  # sell signal
                    success = await self.portfolio_tracker.send_sell_signal(
                        symbol=symbol,
                        selling_price=float(price)
                    )

                if not success:
                    logger.error(f"Failed to send {signal_type} signal to portfolio tracker for {symbol}")
                    return False

                # Send Telegram notification
                signal_emoji = "🔔" if signal_type.lower() == 'buy' else "💰"
                message = (
                    f"{signal_emoji} <b>{signal_type.upper()} Signal Generated!</b>\n\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"Target Price: ${target_price:.2f}\n"
                    f"Timeframe: {timeframe}\n"
                    f"Sentiment Score: {sentiment_score:.2f}\n"
                )
                
                if entry_date and target_date:
                    message += (
                        f"Entry Date: {entry_date.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                        f"Target Date: {target_date.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                    )
                
                message += f"Reason: {reason}\n\n"
                
                # Add news article URL for buy signals
                if signal_type.lower() == 'buy' and hasattr(self, '_current_best_article') and self._current_best_article:
                    article_url = self._current_best_article['article'].get('link', '')
                    if article_url:
                        message += f"Based on news article:\n{article_url}\n\n"
                
                message += (
                    "View details at:\n"
                    "https://portfolio-tracker-rough-dawn-5271.fly.dev"
                )
                
                await self.send_telegram_alert(message)
                return True
                
            except Exception as e:
                logger.error(f"Error in send_signal for {symbol}: {str(e)}", exc_info=True)
                return False

    async def _init_telegram_bot(self):
        """Initialize or reinitialize the Telegram bot"""
        try:
            bot = telegram.Bot(token=self.telegram_token)
            # Test the connection
            await bot.get_me()
            return bot
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {str(e)}")
            raise

    async def send_telegram_alert(self, message: str) -> None:
        """Send alert message via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram bot or chat ID not configured")
            return

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                bot = telegram.Bot(token=self.telegram_token)
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

    async def poll_telegram_updates(self):
        """Poll for Telegram updates"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured, polling disabled")
            return

        async with self._polling_lock:  # Use lock to ensure single instance
            offset = None
            retry_delay = 5
            max_retries = 3

            while self._is_polling:
                for attempt in range(max_retries):
                    try:
                        bot = await self._init_telegram_bot()
                        updates = await bot.get_updates(
                            offset=offset,
                            timeout=30,
                            allowed_updates=['message'],
                            limit=100
                        )
                        
                        if updates:
                            for update in updates:
                                if update.message and update.message.text:
                                    command = update.message.text
                                    chat_id = update.message.chat.id
                                    logger.info(f"Received command: {command} from chat_id: {chat_id}")

                                    try:
                                        if command == '/start' or command == '/help':
                                            await self.send_help_message(chat_id)
                                        elif command == '/portfolio':
                                            await self.send_portfolio_status(chat_id)
                                    except Exception as cmd_error:
                                        logger.error(f"Error handling command {command}: {str(cmd_error)}")

                                # Update offset after processing each update
                                offset = update.update_id + 1
                                logger.debug(f"Updated offset to {offset}")

                        # Small delay between polling requests
                        await asyncio.sleep(1)
                        break  # Break the retry loop if successful
                        
                    except Exception as e:
                        logger.error(f"Error polling telegram updates (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt < max_retries - 1:
                            logger.info(f"Waiting {retry_delay} seconds before retry...")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error("Max retries reached, waiting longer before next attempt")
                            await asyncio.sleep(30)  # Longer delay after all retries fail

    async def send_help_message(self, chat_id):
        """Send help message"""
        try:
            help_message = (
                "🤖 <b>Stock Monitor Bot Commands</b>\n\n"
                "📝 <b>Available Commands:</b>\n\n"
                "/help - Show this help message\n"
                "Get list of all available commands and their descriptions\n\n"
                "/portfolio - View current positions\n"
                "See your portfolio at the tracker website\n\n"
                "ℹ️ <b>About Alerts:</b>\n"
                "• Buy and sell signals are sent automatically\n"
                "• Signals are based on:\n"
                "  - ML predictions 🤖\n"
                "  - Sentiment analysis 📊\n"
                "  - Technical indicators 📈\n\n"
                "📊 <b>Portfolio Tracking:</b>\n"
                "• View your portfolio at: https://portfolio-tracker-rough-dawn-5271.fly.dev\n"
                "• All trades are automatically recorded\n"
                "• Real-time updates and analytics"
            )

            bot = await self._init_telegram_bot()
            await bot.send_message(
                chat_id=chat_id,
                text=help_message,
                parse_mode='HTML'
            )
            logger.info("Help message sent successfully")
        except Exception as e:
            logger.error(f"Error sending help message: {str(e)}")

    async def send_portfolio_status(self, chat_id):
        """Send portfolio status"""
        try:
            message = (
                "📊 <b>Portfolio Status</b>\n\n"
                "View your portfolio and trade history at:\n"
                "https://portfolio-tracker-rough-dawn-5271.fly.dev\n\n"
                "Features:\n"
                "• Real-time position tracking\n"
                "• Performance analytics\n"
                "• Historical trade data\n"
                "• P/L tracking"
            )

            bot = await self._init_telegram_bot()
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info("Portfolio status sent successfully")
        except Exception as e:
            logger.error(f"Error sending portfolio status: {str(e)}")

    async def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            await self.portfolio_tracker.close()
            logger.info("Portfolio tracker session closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        self._is_polling = False  # Stop polling when object is deleted
        if hasattr(self, 'portfolio_tracker'):
            asyncio.create_task(self.cleanup())

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
            logger.info(f"Entry Price: ${entry_price}")
            logger.info(f"Target Price: ${target_price}")
            logger.info(f"Entry Date: {entry_date.isoformat()}")
            logger.info(f"Target Date: {target_date.isoformat()}")
            
            # Ensure portfolio tracker is initialized
            if not hasattr(self, 'portfolio_tracker') or self.portfolio_tracker is None:
                logger.info("Initializing portfolio tracker service")
                self.portfolio_tracker = PortfolioTrackerService()
            
            # Send buy signal
            success = await self.portfolio_tracker.send_buy_signal(
            symbol=symbol,
                entry_price=entry_price,
                target_price=target_price,
                entry_date=entry_date.isoformat(),
                target_date=target_date.isoformat()
            )
            
            if success:
                logger.info("✅ Buy signal sent successfully")
            else:
                logger.error("❌ Failed to send buy signal")
                
            logger.info("=== Portfolio Tracker Test Completed ===")
            
        except Exception as e:
            logger.error(f"Portfolio tracker test failed: {str(e)}", exc_info=True)

    async def periodic_model_check(self):
            """Periodically check and reload models"""
            while True:
                try:
                    # Check for model updates
                    self.model_manager.check_and_reload_models()
                    
                    # Update local references to models
                    self.vectorizers = self.model_manager.vectorizers
                    self.models = {
                        '1h': self.model_manager.models['1h'],
                        '1wk': self.model_manager.models['1wk'],
                        '1mo': self.model_manager.models['1mo']
                    }
                    
                    # Wait before next check
                    await asyncio.sleep(300)  # Check every 5 minutes
                
                except Exception as e:
                    logger.error(f"Error in periodic model check: {e}")
                    await asyncio.sleep(300)  # Wait before retrying

    async def _calculate_predictions(self, symbol: str, articles: List[Dict], price_data: Dict) -> Dict:
        """Calculate predictions with semantic pattern matching"""
        try:
            best_prediction = None
            best_article = None
            best_timeframe = None
            prediction_strength = 0.0
            sentiment_score = 0.0
            should_buy = False
            target_price = None

            current_price = price_data["current_price"]

            for article_data in articles:
                article = article_data['article']
                sentiment = article_data['sentiment']
                content = article.get('title', '') + ' ' + (article.get('description', '') or '')

                # Get embedding for semantic analysis
                embedding = self.semantic_analyzer.get_embedding(content)
                if embedding is None:
                    continue

                # Find similar patterns and their impacts
                similar_patterns = self.semantic_analyzer.find_similar_news(embedding)
                if not similar_patterns:
                    continue

                # Calculate weighted predictions for each timeframe
                for timeframe in ['1h', '1wk', '1mo']:
                    # Get ML model prediction
                    ml_pred = article_data['predictions'].get(timeframe, 0.0)

                    # Get semantic pattern prediction
                    pattern_pred = self.semantic_analyzer.get_semantic_impact_prediction(content, timeframe)
                    if pattern_pred is None:
                        pattern_pred = 0.0

                    # Combine predictions with weights
                    combined_pred = (ml_pred * 0.6) + (pattern_pred * 0.4)
                    
                    # Log predictions
                    logger.info(f"{symbol} {timeframe} predictions:")
                    logger.info(f"  ML: {ml_pred:.2f}%")
                    logger.info(f"  Pattern: {pattern_pred:.2f}%")
                    logger.info(f"  Combined: {combined_pred:.2f}%")

                    # Update best prediction if this is stronger
                    threshold = self.min_price_changes[timeframe]
                    if combined_pred > threshold and (best_prediction is None or combined_pred > best_prediction):
                        best_prediction = combined_pred
                        best_article = article_data
                        best_timeframe = timeframe
                        prediction_strength = combined_pred
                        sentiment_score = sentiment['score']

            if best_prediction is not None:
                should_buy = True
                # Calculate target price based on prediction
                target_price = current_price * (1 + (best_prediction / 100))
                
                # Log cluster information if available
                if hasattr(self.semantic_analyzer, 'news_clusters'):
                    for cluster_id, indices in self.semantic_analyzer.news_clusters.items():
                        impact_stats = self.semantic_analyzer.analyze_cluster_impact(cluster_id, best_timeframe)
                        if impact_stats:
                            logger.info(f"Cluster {cluster_id} stats for {best_timeframe}:")
                            logger.info(f"  Mean impact: {impact_stats['mean']:.2f}%")
                            logger.info(f"  Std dev: {impact_stats['std']:.2f}%")
                            logger.info(f"  Sample size: {impact_stats['count']}")

            return {
                "should_buy": should_buy,
                "best_timeframe": best_timeframe,
                "prediction_strength": prediction_strength,
                "sentiment_score": sentiment_score,
                "target_price": target_price,
                "best_article": best_article
            }

        except Exception as e:
            logger.error(f"Error calculating predictions: {str(e)}")
            return {
                "should_buy": False,
                "best_timeframe": None,
                "prediction_strength": 0.0,
                "sentiment_score": 0.0,
                "target_price": None,
                "best_article": None
            }

    def _is_article_relevant(self, content: str, symbol: str, company_name: str) -> bool:
        """
        Check if the article is relevant to the given stock by looking for the stock symbol
        and company name in the content.
        """
        if not content:
            return False
            
        content = content.lower()
        search_terms = set()
        
        # Add stock symbol variations
        search_terms.add(symbol.lower())
        search_terms.add(f"${symbol.lower()}")
        
        # Add company name variations if available
        if company_name:
            company_name = company_name.lower()
            search_terms.add(company_name)
            
            # Add common company suffixes
            for suffix in [' inc', ' corp', ' ltd', ' llc', ' company']:
                if company_name.endswith(suffix):
                    search_terms.add(company_name[:-len(suffix)])
        
        # Check if any search term appears in the content
        mentions = sum(1 for term in search_terms if term in content)
        
        # Require at least 2 mentions for relevance
        return mentions >= 2

    def analyze_trading_signal(self, news_item, timeframe='1wk'):
        """
        Analyze news item for trading signals
        Returns: dict with signal analysis or None if no clear signal
        """
        try:
            # Get detailed prediction
            prediction = self.model_manager.predict_with_details(news_item['text'], timeframe)
            if not prediction:
                logger.warning(f"Could not get prediction for news item: {news_item['symbol']}")
                return None
                
            # Initialize signal analysis
            signal_analysis = {
                'symbol': news_item['symbol'],
                'timestamp': news_item['timestamp'],
                'signal': 'HOLD',  # Default signal
                'confidence': 0.0,
                'predicted_change': 0.0,
                'analysis_components': {}
            }
            
            # 1. Check base prediction confidence
            if prediction['confidence_score'] < self.min_confidence_threshold:
                logger.info(f"Low prediction confidence for {news_item['symbol']}: {prediction['confidence_score']:.2f}")
                return signal_analysis
            
            # 2. Analyze sentiment
            sentiment = prediction['sentiment_analysis']
            if sentiment['confidence'] < self.min_sentiment_confidence:
                logger.info(f"Low sentiment confidence for {news_item['symbol']}: {sentiment['confidence']:.2f}")
                return signal_analysis
            
            # 3. Check cluster analysis
            cluster_info = prediction['cluster_analysis']
            cluster_valid = False
            if cluster_info and cluster_info['similarity_score'] > self.min_cluster_similarity:
                cluster_stats = cluster_info['cluster_stats']
                cluster_valid = (
                    cluster_stats['size'] >= 3 and  # At least 3 similar articles
                    cluster_stats['std_price_change'] / abs(cluster_stats['avg_price_change']) < 0.5  # Consistent impact
                )
            
            # 4. Calculate combined signal strength
            predicted_change = prediction['price_change_prediction']
            sentiment_score = sentiment['scores']['positive'] - sentiment['scores']['negative']
            
            # Combine all components for final analysis
            signal_strength = 0.0
            signal_components = {
                'model_prediction': {
                    'value': predicted_change,
                    'confidence': prediction['confidence_score'],
                    'weight': 0.4
                },
                'sentiment': {
                    'value': sentiment_score,
                    'confidence': sentiment['confidence'],
                    'weight': 0.3
                },
                'cluster': {
                    'value': cluster_info['cluster_stats']['avg_price_change'] if cluster_valid else 0,
                    'confidence': cluster_info['similarity_score'] if cluster_valid else 0,
                    'weight': 0.3
                } if cluster_valid else None
            }
            
            # Calculate weighted signal strength
            total_weight = 0
            for component_name, component in signal_components.items():
                if component is not None:
                    weighted_value = (
                        component['value'] * 
                        component['confidence'] * 
                        component['weight']
                    )
                    signal_strength += weighted_value
                    total_weight += component['weight']
            
            if total_weight > 0:
                signal_strength = signal_strength / total_weight
            
            # 5. Make final decision
            signal_analysis.update({
                'predicted_change': signal_strength,
                'confidence': prediction['confidence_score'],
                'analysis_components': {
                    'model_prediction': predicted_change,
                    'sentiment_score': sentiment_score,
                    'cluster_impact': cluster_info['cluster_stats']['avg_price_change'] if cluster_valid else None,
                    'signal_strength': signal_strength
                }
            })
            
            # Determine signal based on strength and confidence
            if abs(signal_strength) >= self.min_price_changes[timeframe]:
                if signal_strength > 0:
                    signal_analysis['signal'] = 'BUY'
                    logger.info(f"BUY signal for {news_item['symbol']} - Predicted change: {signal_strength:.2%}")
                else:
                    signal_analysis['signal'] = 'SELL'
                    logger.info(f"SELL signal for {news_item['symbol']} - Predicted change: {signal_strength:.2%}")
                
                # Log detailed analysis
                logger.info(f"Signal Analysis for {news_item['symbol']}:")
                logger.info(f"Timeframe: {timeframe}")
                logger.info(f"Confidence: {prediction['confidence_score']:.2f}")
                logger.info(f"Model Prediction: {predicted_change:.2%}")
                logger.info(f"Sentiment Score: {sentiment_score:.2f}")
                if cluster_valid:
                    logger.info(f"Cluster Impact: {cluster_info['cluster_stats']['avg_price_change']:.2%}")
                logger.info(f"Final Signal Strength: {signal_strength:.2%}")
            
            return signal_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trading signal for {news_item['symbol']}: {str(e)}")
            return None
    
    def filter_strong_signals(self, news_items, timeframe='1wk'):
        """
        Filter news items to find the strongest trading signals
        Returns: List of items with strong buy/sell signals
        """
        strong_signals = []
        
        for item in news_items:
            signal_analysis = self.analyze_trading_signal(item, timeframe)
            if signal_analysis and signal_analysis['signal'] != 'HOLD':
                strong_signals.append(signal_analysis)
        
        # Sort by confidence * predicted_change
        strong_signals.sort(
            key=lambda x: abs(x['predicted_change'] * x['confidence']),
            reverse=True
        )
        
        return strong_signals
    
    def get_buy_recommendations(self, news_items, max_positions=5):
        """
        Get top buy recommendations from news items
        Returns: List of recommended positions with confidence levels
        """
        # First check if models are up to date
        if not self.model_manager.check_and_reload_models():
            logger.error("Failed to check/reload models")
            return []
            
        # Analyze both timeframes
        signals_1wk = self.filter_strong_signals(news_items, '1wk')
        signals_1mo = self.filter_strong_signals(news_items, '1mo')
        
        # Process signals by symbol
        combined_signals = {}
        
        # Process all signals and keep the strongest one for each symbol
        for signal in signals_1wk + signals_1mo:
            if signal['signal'] != 'BUY':
                continue
                
            symbol = signal['symbol']
            timeframe = signal.get('timeframe', '1wk')  # Default to 1wk if not specified
            score = signal['predicted_change'] * signal['confidence']
            
            # Check if prediction meets minimum threshold for its timeframe
            if abs(signal['predicted_change']) < self.min_price_changes[timeframe]:
                continue
                
            if symbol not in combined_signals:
                # First signal for this symbol
                combined_signals[symbol] = {
                    'symbol': symbol,
                    'signal': signal,
                    'score': score
                }
            else:
                # Compare with existing signal
                existing_score = combined_signals[symbol]['score']
                if score > existing_score:
                    # Replace with stronger signal
                    combined_signals[symbol] = {
                        'symbol': symbol,
                        'signal': signal,
                        'score': score
                    }
        
        # Convert to list and sort by score
        recommendations = list(combined_signals.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top N recommendations
        top_recommendations = recommendations[:max_positions]
        
        # Log recommendations
        logger.info("\n" + "="*50)
        logger.info("TOP BUY RECOMMENDATIONS")
        logger.info("="*50)
        
        for rec in top_recommendations:
            signal = rec['signal']
            logger.info(f"\nSymbol: {rec['symbol']}")
            logger.info(f"Timeframe: {signal.get('timeframe', '1wk')}")
            logger.info(f"Predicted Change: {signal['predicted_change']:.2%}")
            logger.info(f"Confidence: {signal['confidence']:.2f}")
            logger.info(f"Score: {rec['score']:.2%}")
            
            # Log analysis components
            components = signal['analysis_components']
            logger.info("Analysis Components:")
            logger.info(f"  Model Prediction: {components['model_prediction']:.2%}")
            logger.info(f"  Sentiment Score: {components['sentiment_score']:.2f}")
            if components.get('cluster_impact'):
                logger.info(f"  Cluster Impact: {components['cluster_impact']:.2%}")
        
        logger.info("="*50)
        
        return top_recommendations
    
    async def process_news_update(self, news_items):
        """Process new news items and generate trading signals"""
        try:
            # Get buy recommendations
            recommendations = self.get_buy_recommendations(news_items)
            
            if not recommendations:
                return []
            
            # Log summary
            logger.info(f"\nProcessed {len(news_items)} news items")
            logger.info(f"Generated {len(recommendations)} buy recommendations")
            
            # Process each recommendation
            for rec in recommendations:
                symbol = rec['symbol']
                signal = rec['signal']
                
                try:
                    # Calculate target price based on predicted change
                    current_price = await self._get_current_price(symbol)
                    if not current_price:
                        logger.error(f"Could not get current price for {symbol}")
                        continue
                    
                    # Calculate entry and target prices
                    entry_price = current_price
                    predicted_change = signal['predicted_change']
                    target_price = entry_price * (1 + predicted_change)
                    
                    # Calculate dates
                    entry_date = datetime.now(tz=pytz.UTC)
                    timeframe = signal.get('timeframe', '1wk')
                    target_date = entry_date + timedelta(
                        days=7 if timeframe == '1wk' else 30
                    )
                    
                    # Send buy signal to portfolio tracker
                    tracking_result = await self.portfolio_tracker.send_buy_signal(
                        symbol=symbol,
                        entry_price=entry_price,
                        target_price=target_price,
                        entry_date=entry_date.isoformat(),
                        target_date=target_date.isoformat()
                    )
                    
                    if tracking_result:
                        # Update position with sentiment and confidence scores
                        await self.portfolio_tracker.update_position(
                            symbol=symbol,
                            target_price=target_price,
                            sentiment_score=signal['analysis_components']['sentiment_score'],
                            confidence_score=signal['confidence']
                        )
                        
                        logger.info(f"Successfully added position for {symbol}")
                        logger.info(f"Entry Price: ${entry_price:.2f}")
                        logger.info(f"Target Price: ${target_price:.2f}")
                        logger.info(f"Predicted Change: {predicted_change:.2%}")
                        logger.info(f"Timeframe: {timeframe}")
                        logger.info(f"Score: {rec['score']:.2%}")
                    else:
                        logger.warning(f"Failed to send buy signal for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing recommendation for {symbol}: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing news update: {str(e)}")
            return []
    
    async def _get_current_price(self, symbol):
        """Get current price for a symbol using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            current = ticker.history(period='1d')
            if not current.empty:
                return current['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

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

async def main():
    logger.info("🚀 Market Monitor Starting...")
    logger.info("Initializing ML models and market data...")
    monitor = RealTimeMonitor()
    
    # Run the portfolio tracker test first
    logger.info("Running portfolio tracker test...")
    await monitor.test_portfolio_tracker()
    
    # Then continue with normal operation
    symbols = get_all_symbols()
    await monitor.start(symbols)

if __name__ == "__main__":
    asyncio.run(main())




