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

# Third-party imports
import aiohttp
import yfinance as yf
import joblib
import pandas as pd
import pytz
from telegram import Update
from telegram.ext import Application, ContextTypes
from telegram.error import TelegramError

# Local application imports
from app.services.news_service import NewsService
from app.utils.sp500 import get_all_symbols
import torch
from app.training.market_ml_trainer import FinBERTSentimentAnalyzer
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
        self.model_paths = {
            'vectorizer': 'app/models/vectorizer.joblib',
            '1h': 'app/models/market_model_1h.joblib',
            '1wk': 'app/models/market_model_1wk.joblib',
            '1mo': 'app/models/market_model_1mo.joblib'
        }
        self.last_modified_times = {}
        self.models = {}
        self.load_initial_models()
    
    def load_initial_models(self):
        for name, path in self.model_paths.items():
            try:
                self.models[name] = joblib.load(path)
                self.last_modified_times[name] = os.path.getmtime(path)
                logger.info(f"Loaded initial {name} model")
            except Exception as e:
                logger.error(f"Error loading initial {name} model: {e}")

    def check_and_reload_models(self):
        """Check if models have been updated and reload if necessary"""
        try:
            for name, path in self.model_paths.items():
                current_mod_time = os.path.getmtime(path)
                
                if current_mod_time > self.last_modified_times.get(name, 0):
                    try:
                        new_model = joblib.load(path)
                        self.models[name] = new_model
                        self.last_modified_times[name] = current_mod_time
                        
                        logger.info(f"🔄 Reloaded {name} model")
                        logger.info(f"Model last modified: {datetime.fromtimestamp(current_mod_time)}")
                    except Exception as e:
                        logger.error(f"Error reloading {name} model: {e}")
        except Exception as e:
            logger.error(f"Error in model check: {e}")

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
        self.news_aggregator = NewsAggregator()
        
        # Initialize portfolio tracker
        self.portfolio_tracker = PortfolioTrackerService()
        logger.info("Portfolio tracker service initialized")
        
        self.stop_loss_percentage = 5.0
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.sentiment_weight = 0.3

        self.model_manager = ModelManager()
        
        # Use models from model manager
        self.vectorizer = self.model_manager.models['vectorizer']
        self.models = {
            '1h': self.model_manager.models['1h'],
            '1wk': self.model_manager.models['1wk'],
            '1mo': self.model_manager.models['1mo']
        }
        
        # Initialize Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not found in environment variables")
        else:
            logger.info("Telegram credentials loaded successfully")
                
        self.news_service = NewsService()
        self.processed_news = set()

        self.thresholds = {
            '1h': 5.0,
            '1wk': 10.0,
            '1mo': 20.0
        }

        self._polling_lock = asyncio.Lock()
        self._is_polling = False

    def _pad_features(self, X):
        """Pad or truncate features to match model input size"""
        try:
            # Expected number of features from the model
            expected_features = 84953
            current_features = X.shape[1]
            logger.info(f"Input matrix shape: {X.shape}")

            if current_features < expected_features:
                # Create zero matrix for padding
                padding = scipy.sparse.csr_matrix((X.shape[0], expected_features - current_features))
                # Horizontally stack with original features
                X_padded = scipy.sparse.hstack([X, padding])
                logger.info(f"Padded matrix shape: {X_padded.shape}")
                return X_padded
            elif current_features > expected_features:
                # Truncate to expected size
                X_truncated = X[:, :expected_features]
                logger.info(f"Truncated matrix shape: {X_truncated.shape}")
                return X_truncated
            else:
                return X
        except Exception as e:
            logger.error(f"Feature padding error: {str(e)}")
            logger.error(f"Input matrix shape: {X.shape}")
            raise

    def predict_with_sentiment(self, text, timeframe):
        """
        Make a prediction integrating continuous FinBERT sentiment scores
        """
        try:
            # Vectorize text
            X_tfidf = self.vectorizer.transform([text])
            
            # Pad features to match expected size
            X_padded = self._pad_features(X_tfidf)
            
            # Base prediction from model
            base_pred = self.models[timeframe].predict(X_padded)[0]
            
            # Analyze sentiment
            sentiment = self.finbert_analyzer.analyze_sentiment(text)
            
            # If sentiment available, adjust prediction
            if sentiment:
                # Get continuous sentiment score (-1 to 1)
                sentiment_score = sentiment['score']
                
                # Calculate multiplier based on continuous score
                # Score of -1 -> multiplier of 0.7
                # Score of 0  -> multiplier of 1.0
                # Score of 1  -> multiplier of 1.3
                base_multiplier = 1.0 + (sentiment_score * 0.3)
                
                # Adjusted prediction
                adjusted_pred = base_pred * (
                    1 + (base_multiplier - 1) * self.sentiment_weight
                )
                
                # Logging for transparency
                logger.info(f"Prediction for {timeframe}:")
                logger.info(f"Base Prediction: {base_pred:.2f}%")
                logger.info(f"Sentiment Score: {sentiment_score:.2f}")
                logger.info(f"Sentiment Multiplier: {base_multiplier:.2f}")
                logger.info(f"Adjusted Prediction: {adjusted_pred:.2f}%")
                
                return adjusted_pred
            
            return base_pred
            
        except Exception as e:
            logger.error(f"Error in sentiment-integrated prediction: {e}")
            return base_pred
    
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

    async def analyze_and_notify(self, symbol: str, articles: List[Dict], price_data: Dict) -> None:
        try:
            # Calculate predictions and check thresholds
            prediction_result = await self._calculate_predictions(symbol, articles, price_data)
            
            if prediction_result["should_buy"]:
                # Check if target price has already been reached
                current_price = price_data["current_price"]
                target_price = prediction_result["target_price"]
                
                # Calculate how much of the predicted move has already happened
                predicted_move = ((target_price - current_price) / current_price) * 100
                
                # If more than 30% of the predicted move has already happened, skip this signal
                if predicted_move <= 0 or predicted_move * 0.3 <= 0:
                    logger.info(f"Skipping buy signal for {symbol} - Target price has already been reached or invalid")
                    return
                
                logger.info("🟢 BUY SIGNAL DETECTED")
                logger.info(f"Best Timeframe: {prediction_result['best_timeframe']}")
                logger.info(f"Prediction Strength: {prediction_result['prediction_strength']:.2f}%")
                
                # Get the most significant article's publish date
                best_article = prediction_result["best_article"]
                entry_date = datetime.fromtimestamp(best_article['article']['providerPublishTime'], tz=pytz.UTC)
                
                # Store the best article for the signal
                self._current_best_article = best_article
                
                # Calculate target date based on timeframe
                timeframe_deltas = {
                    '1h': timedelta(hours=1),
                    '1wk': timedelta(weeks=1),
                    '1mo': timedelta(days=30)
                }
                target_date = entry_date + timeframe_deltas[prediction_result['best_timeframe']]
                
                logger.info(f"Using article published at {entry_date} as entry date")
                logger.info(f"Target date set to {target_date} based on {prediction_result['best_timeframe']} timeframe")
                
                await self.send_signal(
                    signal_type="buy",
                    symbol=symbol,
                    price=current_price,
                    target_price=target_price,
                    sentiment_score=prediction_result["sentiment_score"],
                    timeframe=prediction_result["best_timeframe"],
                    reason=f"ML prediction: {prediction_result['prediction_strength']:.1f}% upside potential",
                    entry_date=entry_date,
                    target_date=target_date
                )
                
                # Clear the stored article after sending the signal
                self._current_best_article = None
                
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")

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
                    self.vectorizer = self.model_manager.models['vectorizer']
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
    async def start(self, symbols: list[str]):
        """Start the monitoring process"""
        try:
            startup_message = (
                "🚀 <b>Stock Monitor Activated</b>\n\n"
                f"📊 Monitoring {len(symbols)} stocks\n"
                f"🕒 Started at: {datetime.now(tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                "🔍 Initializing market monitoring process..."
            )
            await self.send_telegram_alert(startup_message)
        except Exception as alert_error:
            logger.error(f"Failed to send startup Telegram alert: {alert_error}")

        logger.info(f"Starting monitoring for {len(symbols)} symbols...")
        
        # Initialize polling lock if not exists
        if not hasattr(self, '_polling_lock'):
            self._polling_lock = asyncio.Lock()
        
        self._is_polling = True
        
        # Start polling in background
        polling_task = asyncio.create_task(self.poll_telegram_updates())
        logger.info("Telegram polling started")

        model_check_task = asyncio.create_task(self.periodic_model_check())
   
        try:
            while True:
                try:
                    logger.info("Starting new monitoring cycle...")
                    
                    # Process stocks in batches for better resource management
                    batch_size = 50
                    total_symbols = len(symbols)
                    
                    for batch_start in range(0, total_symbols, batch_size):
                        batch = symbols[batch_start:batch_start+batch_size]
                        batch_number = batch_start // batch_size + 1
                        total_batches = (total_symbols + batch_size - 1) // batch_size
                        
                        logger.info(f"Processing Batch {batch_number}/{total_batches}")
                        
                        async def process_symbols():
                            sem = asyncio.Semaphore(50)
                            
                            async def bounded_monitor(symbol):
                                async with sem:
                                    return await self.monitor_stock_with_tracking(symbol, batch_number, total_batches)
                            
                            tasks = [bounded_monitor(symbol) for symbol in batch]
                            return await asyncio.gather(*tasks)
                
                        await process_symbols()
                        logger.info(f"Completed Batch {batch_number}/{total_batches}")
                        await asyncio.sleep(1)
                    
                    logger.info("Completed full monitoring cycle. Waiting 5 minutes...")
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {str(e)}")
                    await asyncio.sleep(300)
        finally:
            self._is_polling = False
            await polling_task
            await model_check_task

    async def monitor_stock_with_tracking(self, symbol: str, batch_number: int, total_batches: int):
        """Wrapper method to add tracking and logging to monitor_stock"""
        start_time = time.time()
        try:
            logger.info(f"🔍 Processing {symbol} (Batch {batch_number}/{total_batches})")
            await self.monitor_stock(symbol)
            duration = time.time() - start_time
            logger.info(f"✅ Completed {symbol} in {duration:.2f} seconds (Batch {batch_number}/{total_batches})")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Failed processing {symbol} in {duration:.2f} seconds: {str(e)}")

    async def monitor_stock(self, symbol: str):
        try:
            # Change from one week to 24 hours
            one_day_ago = datetime.now(tz=pytz.UTC) - timedelta(hours=24)

            logger.info(f"🔄 Processing stock: {symbol}")
            stock = yf.Ticker(symbol)
            
            try:
                prices = stock.history(period="3mo", interval="1h")
                if prices.empty:
                    logger.warning(f"Skipping {symbol}: No recent price data available")
                    return
                    
                # Calculate current price from the most recent data
                current_price = float(prices['Close'].iloc[-1])
                price_data = {
                    "current_price": current_price,
                    "history": prices
                }
                logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {str(e)}")
                return

            # Filter news when fetching - changed to 24 hours
            news = [
                article for article in stock.news 
                if datetime.fromtimestamp(article['providerPublishTime'], tz=pytz.UTC) >= one_day_ago
            ]

            if not news:
                logger.info(f"ℹ️ No recent news (last 24 hours) found for {symbol}")
                return

            # Limit to top 10 recent news articles
            news = news[:10]
            all_predictions = []
            processed_urls = set()  # Track processed URLs to avoid duplicates

            for article in news:
                url = article.get('link')
            
                # Skip if URL already processed
                if url in processed_urls:
                    continue

                news_id = f"{symbol}_{article['link']}"
                if news_id in self.processed_news:
                    continue

                try:
                    # Get full article text
                    url = article.get('link')
                    full_text = None
                    
                    if url:
                        logger.info(f"Fetching full article from: {url}")
                        full_text = self.get_full_article_text(url)
                        
                        if full_text:
                            logger.info(f"Retrieved article length: {len(full_text)} characters")
                    
                    # Combine title and full text
                    content = article.get('title', '')
                    if full_text:
                        content = f"{content}\n\n{full_text}"

                    # Check if article is relevant to this stock
                    if not self._is_article_relevant(content, symbol, stock.info.get('longName', '')):
                        logger.info(f"Skipping article - not relevant to {symbol}")
                        continue

                    # Make predictions for this specific article
                    article_predictions = {}
                    
                    # Get TF-IDF features only
                    X_tfidf = self.vectorizer.transform([content])
                    
                    # Get sentiment analysis from FinBERT (for later adjustment)
                    sentiment = self.finbert_analyzer.analyze_sentiment(content)
                    
                    predictions_log = []  # Collect all predictions for single log message
                    
                    for timeframe, model in self.models.items():
                        try:
                            # Create sentiment features array (same as in training)
                            if sentiment:
                                sentiment_features = np.array([
                                    sentiment['score'],  # Base score
                                    sentiment['confidence'],  # Confidence
                                    sentiment['probabilities']['positive'],
                                    sentiment['probabilities']['negative'],
                                    sentiment['probabilities']['neutral']
                                ]).reshape(1, -1)
                            else:
                                # If no sentiment, use zeros
                                sentiment_features = np.zeros((1, 5))
                            
                            # Convert to sparse matrix
                            sentiment_sparse = scipy.sparse.csr_matrix(sentiment_features)
                            
                            # Combine features exactly as done in training
                            X = scipy.sparse.hstack([X_tfidf, sentiment_sparse])
                            
                            # Get prediction using combined features
                            pred = model.predict(X)[0]
                            
                            # Log prediction details
                            logger.info(f"Prediction for {timeframe}: {pred:.2f}%")
                            if sentiment:
                                logger.info(f"Sentiment Score: {sentiment['score']:.2f}")
                                logger.info(f"Confidence: {sentiment['confidence']:.2f}")
                            
                            article_predictions[timeframe] = pred
                            predictions_log.append(f"{timeframe}: {pred:.2f}%")
                        
                        except Exception as e:
                            logger.error(f"Error in prediction for {timeframe}: {e}")
                            continue
                    
                    # Log all predictions in a single message
                    if predictions_log:
                        logger.info(f"Predictions for {symbol}: " + ", ".join(predictions_log))
                    
                    all_predictions.append({
                        'article': article,
                        'predictions': article_predictions,
                        'sentiment': sentiment
                    })
                    
                    self.processed_news.add(news_id)
                    processed_urls.add(url)
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                    continue
            
            if all_predictions:
                await self.analyze_and_notify(
                    symbol=symbol,
                    articles=all_predictions,
                    price_data=price_data
                )
        
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")
            return

    async def update_positions(self):
        """Update and check all positions for selling conditions"""
        logger.info("Updating portfolio positions...")
        current_time = datetime.now(tz=pytz.UTC)

        try:
            # Get current positions from portfolio tracker
            positions = await self.portfolio_tracker.get_positions()
            if not positions:
                logger.info("No active positions to update")
                return

            for position in positions:
                try:
                    symbol = position['symbol']
                    entry_price = float(position['entry_price'])
                    target_price = float(position['target_price'])
                    target_date = datetime.fromisoformat(position['target_date'])
                    timeframe = position.get('timeframe', '1wk')  # Default to 1wk if not specified

                    # Get current price
                    stock = yf.Ticker(symbol)
                    current_price = float(stock.info.get('regularMarketPrice', 0))
                    
                    if current_price <= 0:
                        logger.warning(f"Could not get valid price for {symbol}")
                        continue

                    price_change_percent = ((current_price - entry_price) / entry_price) * 100
                    logger.info(f"Checking {symbol} - Current: ${current_price:.2f}, Entry: ${entry_price:.2f}, Target: ${target_price:.2f}, Change: {price_change_percent:.2f}%")

                    sell_signal = None
                    
                    # Check selling conditions
                    if current_price >= target_price:
                        sell_signal = {
                            "reason": f"🎯 Target price ${target_price:.2f} achieved! ({price_change_percent:.2f}% gain)",
                            "condition": "TARGET_PRICE"
                        }
                    elif price_change_percent <= -self.stop_loss_percentage:
                        sell_signal = {
                            "reason": f"⚠️ Stop loss triggered: Price dropped {abs(price_change_percent):.2f}%",
                            "condition": "STOP_LOSS"
                        }
                    elif current_time >= target_date:
                        if price_change_percent > 0:
                            reason = f"📅 Target date reached with {price_change_percent:.2f}% profit"
                        else:
                            reason = f"📅 Target date reached with {abs(price_change_percent):.2f}% loss"
                        sell_signal = {
                            "reason": reason,
                            "condition": "TARGET_DATE"
                        }

                    if sell_signal:
                        logger.info(f"Selling condition met for {symbol}: {sell_signal['reason']}")
                        
                        # Send sell signal to portfolio tracker
                        success = await self.portfolio_tracker.send_sell_signal(
                            symbol=symbol,
                            selling_price=current_price,
                            sell_condition=sell_signal['condition']
                        )

                        if success:
                            # Send Telegram notification
                            await self.send_signal(
                                signal_type="sell",
                                symbol=symbol,
                                price=current_price,
                                target_price=target_price,
                                sentiment_score=0.0,
                                timeframe=timeframe,
                                reason=sell_signal['reason']
                            )
                            logger.info(f"Successfully closed position for {symbol}")
                        else:
                            logger.error(f"Failed to send sell signal to portfolio tracker for {symbol}")

                except Exception as e:
                    logger.error(f"Error updating position for {symbol}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in update_positions: {str(e)}")

        logger.info("Portfolio update complete")

    def calculate_normalized_score(self, changes):
        """Normalize scores based on timeframe thresholds"""
        normalized_scores = {}
        for timeframe, change in changes.items():
            threshold = self.thresholds[timeframe]
            normalized_score = change / threshold
            normalized_scores[timeframe] = normalized_score
            logger.info(f"{timeframe} Change: {change:.2f}% (Normalized Score: {normalized_score:.2f})")
        return normalized_scores

    
    def calculate_impact(self, prices, publish_time):
        if prices.empty:
            return {'impact': 1, 'scores': None, 'changes': None}

        publish_datetime = datetime.fromtimestamp(publish_time)
        print("Publish Date: ", publish_datetime.strftime('%Y-%m-%d %H:%M:%S'))

        changes = {}
        start_price = None
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
                change = ((end_price - start_price) / start_price) * 100

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
    def process_market_news(self):
        for symbol in self.symbols:
            news_items = self.get_news_for_symbol(symbol)
            for news in news_items:
                news_id = f"{symbol}_{news['id']}"
                if news_id not in self.processed_news:
                    self.analyze_and_send_alert(news, symbol)
                    self.processed_news.add(news_id)
                    
    def run(self):
        while True:
            self.process_market_news()
            print(f"Processed news count: {len(self.processed_news)}")
            time.sleep(300)  # 5-minute break

    def analyze_sentiment(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        return {
            "score": int(prediction),
            "confidence": float(max(probabilities)),
            "label": ["bearish", "neutral", "bullish"][prediction]
        }


    async def _calculate_predictions(self, symbol: str, articles: List[Dict], price_data: Dict) -> Dict:
        """Calculate predictions and determine if buy signal should be generated"""
        try:
            # Initialize result dictionary
            result = {
                "should_buy": False,
                "best_timeframe": None,
                "prediction_strength": 0.0,
                "target_price": 0.0,
                "sentiment_score": 0.0,
                "best_article": None
            }

            # Calculate average sentiment score using continuous scores
            sentiment_scores = []
            for article in articles:
                if 'sentiment' in article and article['sentiment']:
                    sentiment = article['sentiment']
                    if isinstance(sentiment, dict) and 'score' in sentiment:
                        # Use continuous score directly
                        sentiment_scores.append(sentiment['score'])

            result["sentiment_score"] = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Find best prediction across timeframes and articles
            best_prediction = 0.0
            best_timeframe = None
            best_article = None

            for article in articles:
                if 'predictions' in article:
                    for timeframe, prediction in article['predictions'].items():
                        if prediction > best_prediction:
                            best_prediction = prediction
                            best_timeframe = timeframe
                            best_article = article

            if best_timeframe and best_prediction and best_article:
                result["best_timeframe"] = best_timeframe
                result["prediction_strength"] = best_prediction
                result["best_article"] = best_article

                # Calculate target price based on prediction
                current_price = price_data.get("current_price", 0)
                if current_price > 0:
                    result["target_price"] = current_price * (1 + best_prediction/100)

                # Check if prediction meets threshold for buy signal
                threshold_met = best_prediction >= self.thresholds.get(best_timeframe, float('inf'))
                # Use continuous sentiment threshold (-1 to 1 scale)
                sentiment_threshold_met = result["sentiment_score"] >= 0.8  # Positive sentiment threshold

                # Generate buy signal if both thresholds are met
                result["should_buy"] = threshold_met and sentiment_threshold_met

            return result

        except Exception as e:
            logger.error(f"Error calculating predictions for {symbol}: {str(e)}")
            return {
                "should_buy": False,
                "best_timeframe": None,
                "prediction_strength": 0.0,
                "target_price": 0.0,
                "sentiment_score": 0.0,
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




