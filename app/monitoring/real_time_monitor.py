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
        """Load all initial models with enhanced error handling"""
        # Load vectorizer first as it's required for other models
        try:
            vectorizer_path = self.model_paths['vectorizer']
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            
            self.models['vectorizer'] = joblib.load(vectorizer_path)
            self.last_modified_times['vectorizer'] = os.path.getmtime(vectorizer_path)
            logger.info("✅ Loaded vectorizer successfully")
            
            # Now load other models
            for name, path in self.model_paths.items():
                if name == 'vectorizer':
                    continue
                    
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found at {path}")
                
                self.models[name] = joblib.load(path)
                self.last_modified_times[name] = os.path.getmtime(path)
                logger.info(f"✅ Loaded {name} model successfully")
                
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {str(e)}")
            raise  # Re-raise the exception as this is critical

    def check_and_reload_models(self):
        """Check if models have been updated and reload if necessary"""
        try:
            for name, path in self.model_paths.items():
                if not os.path.exists(path):
                    logger.error(f"❌ Model file not found at {path}")
                    continue
                    
                current_mod_time = os.path.getmtime(path)
                
                if current_mod_time > self.last_modified_times.get(name, 0):
                    try:
                        new_model = joblib.load(path)
                        self.models[name] = new_model
                        self.last_modified_times[name] = current_mod_time
                        
                        logger.info(f"🔄 Reloaded {name} model")
                        logger.info(f"Model last modified: {datetime.fromtimestamp(current_mod_time)}")
                    except Exception as e:
                        logger.error(f"❌ Error reloading {name} model: {str(e)}")
                        
        except Exception as e:
            logger.error(f"❌ Error in model check: {str(e)}")

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
        
        # Thresholds for significant price movements
        self.thresholds = {
            '1h': 5.0,   # 5% threshold
            '1wk': 10.0,  # 10% threshold
            '1mo': 20.0   # 20% threshold
        }
        
        # Portfolio check interval (in minutes)
        self.portfolio_check_interval = 5
        
        # Minimum requirements for signal generation
        self.signal_criteria = {
            'min_confidence': 0.6,
            'min_alignment_rate': 0.6,
            'min_sentiment_score': 0.3,
            'max_neutral_probability': 0.4
        }

        # Initialize ModelManager first
        self.model_manager = ModelManager()
        logger.info("Model manager initialized")
        
        # Get models from model manager
        if 'vectorizer' not in self.model_manager.models:
            raise ValueError("Vectorizer not found in model manager")
        
        self.vectorizer = self.model_manager.models['vectorizer']
        self.models = {
            '1h': self.model_manager.models['1h'],
            '1wk': self.model_manager.models['1wk'],
            '1mo': self.model_manager.models['1mo']
        }
        
        logger.info("ML models loaded successfully")
        
        # Initialize Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not found in environment variables")
        else:
            logger.info("Telegram credentials loaded successfully")
                
        self.news_service = NewsService()
        self.processed_news = set()

        self._polling_lock = asyncio.Lock()
        self._is_polling = False
        self._portfolio_check_task = None

    def _pad_features(self, X):
        """Pad or truncate features to match model input size with proper normalization"""
        try:
            # Get the expected number of features from the model
            expected_features = 84953
            current_features = X.shape[1]
            logger.info(f"Input matrix shape: {X.shape}")
            logger.info(f"Expected features: {expected_features}, Current features: {current_features}")

            # Normalize the features using L2 norm
            X_normalized = scipy.sparse.csr_matrix(X)
            norm = np.sqrt(X_normalized.multiply(X_normalized).sum(axis=1))
            # Avoid division by zero
            norm[norm == 0] = 1
            X_normalized = scipy.sparse.diags(1/np.array(norm).flatten()) @ X_normalized

            if current_features < expected_features:
                # Create zero matrix for padding
                padding = scipy.sparse.csr_matrix((X.shape[0], expected_features - current_features))
                # Horizontally stack with original features
                X_padded = scipy.sparse.hstack([X_normalized, padding])
            elif current_features > expected_features:
                # Truncate to expected size
                X_padded = X_normalized[:, :expected_features]
            else:
                X_padded = X_normalized

            logger.info(f"Padded matrix shape: {X_padded.shape}")
            return X_padded

        except Exception as e:
            logger.error(f"Feature padding error: {str(e)}")
            raise

    def predict_with_sentiment(self, text, timeframe):
        """
        Make a prediction integrating FinBERT sentiment with significance thresholds
        """
        try:
            # Vectorize text
            X_tfidf = self.vectorizer.transform([text])
            logger.info(f"Original TF-IDF shape: {X_tfidf.shape}")
            
            # Normalize features (L2 norm)
            X_normalized = scipy.sparse.csr_matrix(X_tfidf)
            X_normalized.data = X_normalized.data / np.sqrt(np.sum(X_normalized.data ** 2))
            
            # Pad features if needed
            X_padded = self._pad_features(X_normalized)
            logger.info(f"After padding shape: {X_padded.shape}")
            
            # Make prediction and validate
            base_pred = self.models[timeframe].predict(X_padded)[0]
            
            # Clip prediction to reasonable range (-100% to +100%)
            base_pred = np.clip(base_pred, -100, 100)
            
            # Analyze sentiment
            sentiment = self.finbert_analyzer.analyze_sentiment(text)
            
            # If sentiment available, adjust prediction
            if sentiment:
                # Calculate multiplier using continuous score
                base_multiplier = 1.0 + (sentiment['score'] * 0.3)
                confidence = sentiment['confidence']
                neutral_dampener = 1.0 - sentiment['probabilities']['neutral']
                adjusted_multiplier = 1.0 + (base_multiplier - 1.0) * confidence * neutral_dampener
                pred = base_pred * adjusted_multiplier
                
                # Clip final prediction again
                pred = np.clip(pred, -100, 100)
                
                # Check if prediction meets threshold
                threshold = self.thresholds.get(timeframe, float('inf'))
                meets_threshold = abs(pred) >= threshold
                
                # Logging for transparency
                logger.info(f"\nPrediction Analysis for {timeframe}:")
                logger.info(f"Base Prediction: {base_pred:.2f}%")
                logger.info(f"Sentiment Score: {sentiment['score']:.2f}")
                logger.info(f"Confidence: {confidence:.2f}")
                logger.info(f"Neutral Probability: {sentiment['probabilities']['neutral']:.2f}")
                logger.info(f"Sentiment Multiplier: {adjusted_multiplier:.2f}")
                logger.info(f"Adjusted Prediction: {pred:.2f}%")
                logger.info(f"Threshold ({threshold}%) Met: {meets_threshold}")
                
                if meets_threshold:
                    return pred
                else:
                    logger.info(f"Prediction {pred:.2f}% does not meet {timeframe} threshold of {threshold}%")
                    return 0.0
            
            # Clip and validate final prediction
            base_pred = np.clip(base_pred, -100, 100)
            return base_pred if abs(base_pred) >= self.thresholds.get(timeframe, float('inf')) else 0.0
        
        except Exception as e:
            logger.error(f"Error in sentiment-integrated prediction: {e}")
            return 0.0

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

    async def handle_position_update(self, symbol: str, current_price: float, prediction_result: Dict) -> None:
        """Handle updates to existing positions based on new signals"""
        try:
            # Get current position for this symbol
            position = await self.portfolio_tracker.get_position(symbol)
            if not position:
                return

            entry_price = float(position['entryPrice'])
            target_price = float(position['targetPrice'])
            target_date = datetime.fromisoformat(position['targetDate'])
            current_return = ((current_price - entry_price) / entry_price) * 100
            current_time = datetime.now(tz=pytz.UTC)
            
            # Check for sell conditions
            should_sell = False
            sell_reason = None
            sell_condition = None
            
            # 1. Stop loss hit
            if current_return <= -self.stop_loss_percentage:
                should_sell = True
                sell_reason = f"Stop loss triggered at {current_return:.1f}% loss"
                sell_condition = "STOP_LOSS"
            
            # 2. Target price reached
            elif current_price >= target_price:
                should_sell = True
                sell_reason = f"Target price ${target_price:.2f} reached"
                sell_condition = "TARGET_REACHED"
            
            # 3. Target date reached
            elif current_time >= target_date:
                if current_return > 0:
                    sell_reason = f"Target date reached with {current_return:.1f}% profit"
                else:
                    sell_reason = f"Target date reached with {abs(current_return):.1f}% loss"
                should_sell = True
                sell_condition = "TARGET_DATE"
            
            # 4. Sentiment turned significantly negative
            elif prediction_result['sentiment_score'] < -0.3 and prediction_result['confidence_score'] >= 0.6:
                should_sell = True
                sell_reason = f"Sentiment turned negative: {prediction_result['sentiment_score']:.2f}"
                sell_condition = "PREDICTION_BASED"
            
            # 5. Prediction suggests price reversal
            elif prediction_result['prediction_strength'] < -self.thresholds.get(prediction_result['best_timeframe'], 0):
                should_sell = True
                sell_reason = f"Price reversal predicted: {prediction_result['prediction_strength']:.1f}%"
                sell_condition = "PREDICTION_BASED"

            if should_sell:
                logger.info(f"Selling {symbol} - Reason: {sell_reason}")
                
                # First send sell signal to portfolio tracker
                sell_success = await self.portfolio_tracker.send_sell_signal(
                    symbol=symbol,
                    selling_price=current_price,
                    sell_condition=sell_condition
                )
                
                if sell_success:
                    # Then send notification
                    await self.send_signal(
                        signal_type="sell",
                        symbol=symbol,
                        price=current_price,
                        target_price=0.0,
                        sentiment_score=prediction_result['sentiment_score'],
                        timeframe=prediction_result['best_timeframe'],
                        reason=sell_reason
                    )
                    logger.info(f"Successfully closed position for {symbol}")
                else:
                    logger.error(f"Failed to send sell signal to portfolio tracker for {symbol}")
                return

            # Check for position updates
            if prediction_result['should_buy']:
                # Update target if new prediction is significantly different
                new_target = prediction_result['target_price']
                target_change = abs((new_target - target_price) / target_price) * 100
                
                if target_change >= 5.0:  # Only update if target changes by 5% or more
                    logger.info(f"Updating {symbol} position with new target: ${new_target:.2f}")
                    await self.portfolio_tracker.update_position(
                        symbol=symbol,
                        target_price=new_target,
                        sentiment_score=prediction_result['sentiment_score'],
                        confidence_score=prediction_result['confidence_score']
                    )
                    
                    # Send notification about position update
                    update_message = (
                        f"[UPDATE] Position Update for {symbol}\n"
                        f"New Target: ${new_target:.2f}\n"
                        f"Current Return: {current_return:.1f}%\n"
                        f"Sentiment Score: {prediction_result['sentiment_score']:.2f}\n"
                        f"Confidence Score: {prediction_result['confidence_score']:.2f}"
                    )
                    await self.send_telegram_alert(update_message)
        
        except Exception as e:
            logger.error(f"Error handling position update for {symbol}: {str(e)}")

    async def get_current_price(self, symbol: str, is_market_hours: bool = True) -> Optional[float]:
        """Get current price considering market hours"""
        try:
            stock = yf.Ticker(symbol)
            
            if is_market_hours:
                # During market hours, use regular price
                price = stock.info.get('regularMarketPrice')
                if price:
                    return float(price)
            
            # For pre/post market, try to get the appropriate price
            if stock.info.get('marketState') == 'PRE':
                price = stock.info.get('preMarketPrice')
                if price:
                    return float(price)
            elif stock.info.get('marketState') in ['POST', 'POSTPOST']:
                price = stock.info.get('postMarketPrice')
                if price:
                    return float(price)
            
            # Fallback to regular market price if others not available
            return float(stock.info.get('regularMarketPrice', 0))
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def is_market_hours(self) -> bool:
        """Check if it's currently regular market hours (9:30 AM - 4:00 PM ET, weekdays)"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Create time objects for market open (9:30 AM) and close (4:00 PM)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

    async def analyze_and_notify(self, symbol: str, articles: List[Dict], price_data: Dict) -> None:
        try:
            # Get current price with market hours consideration
            is_market = self.is_market_hours()
            current_price = await self.get_current_price(symbol, is_market)
            if not current_price:
                logger.error(f"Could not get valid price for {symbol}")
                return
                
            # Update price_data with accurate price
            price_data["current_price"] = current_price
            
            # Calculate predictions
            prediction_result = await self._calculate_predictions(symbol, articles, price_data)
            
            # First check if we have an existing position
            await self.handle_position_update(symbol, current_price, prediction_result)
            
            # Only proceed with new buy signal if we don't have an existing position
            position = await self.portfolio_tracker.get_position(symbol)
            if position:
                return
            
            if prediction_result["should_buy"]:
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
                
                # Calculate target date based on timeframe
                timeframe_deltas = {
                    '1h': timedelta(hours=1),
                    '1wk': timedelta(weeks=1),
                    '1mo': timedelta(days=30)
                }
                target_date = entry_date + timeframe_deltas[prediction_result['best_timeframe']]
                
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
            
            message += (
                f"Reason: {reason}\n\n"
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

    async def start_portfolio_checker(self):
        """Start the periodic portfolio checker"""
        logger.info(f"Starting portfolio checker (checking every {self.portfolio_check_interval} minutes)")
        while True:
            try:
                await self.update_positions()
                await asyncio.sleep(self.portfolio_check_interval * 60)  # Convert minutes to seconds
            except Exception as e:
                logger.error(f"Error in portfolio checker: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying if there's an error

    async def start(self, symbols: List[str]):
        """Start the market monitor with portfolio checking"""
        try:
            # Start portfolio checker in the background
            self._portfolio_check_task = asyncio.create_task(self.start_portfolio_checker())
            logger.info("Portfolio checker started successfully")
            
            # Start the main monitoring loop
            while True:
                if not self._is_polling:
                    async with self._polling_lock:
                        self._is_polling = True
                        try:
                            await self.poll_news(symbols)
                        finally:
                            self._is_polling = False
                await asyncio.sleep(60)  # Wait 1 minute between polls
                
        except Exception as e:
            logger.error(f"Error in market monitor: {str(e)}")
        finally:
            # Clean up portfolio checker if it's running
            if self._portfolio_check_task:
                self._portfolio_check_task.cancel()
                try:
                    await self._portfolio_check_task
                except asyncio.CancelledError:
                    pass

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

            # Limit to top 12 recent news articles
            news = news[:12]
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
                     # Analyze sentiment
                    sentiment = self.finbert_analyzer.analyze_sentiment(content)
                
                    # Make predictions for this specific article
                    article_predictions = {}
                    X = self.vectorizer.transform([content])
                
                    # Pad and normalize features BEFORE prediction
                    X_padded = self._pad_features(X)
                    
                    predictions_log = []
                    
                    for timeframe, model in self.models.items():
                        try:
                            # Make prediction and apply sigmoid to bound it
                            raw_pred = model.predict(X_padded)[0]
                            
                            # Apply sigmoid transformation to bound predictions
                            bounded_pred = 100 * (2 / (1 + np.exp(-raw_pred/100)) - 1)
                            
                            if sentiment:
                                # Calculate multiplier using continuous score with dampened effect
                                sentiment_effect = np.clip(sentiment['score'], -0.3, 0.3)
                                confidence = sentiment['confidence']
                                neutral_dampener = 1.0 - sentiment['probabilities']['neutral']
                                
                                # Apply a more conservative adjustment
                                adjusted_pred = bounded_pred * (1 + sentiment_effect * confidence * neutral_dampener)
                                
                                # Final clip to ensure reasonable bounds
                                final_pred = np.clip(adjusted_pred, -100, 100)
                            else:
                                final_pred = np.clip(bounded_pred, -100, 100)
                            
                            article_predictions[timeframe] = final_pred
                            predictions_log.append(f"{timeframe}: {final_pred:.2f}%")
                        
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
                    entry_price = float(position['entryPrice'])
                    target_price = float(position['targetPrice'])
                    target_date = datetime.fromisoformat(position['targetDate'])
                    current_return = ((current_price - entry_price) / entry_price) * 100
                    current_price = float(position['currentPrice'])
                    
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
                            reason = f"📅 Target date reached with {abs(price_change_percent):.1f}% loss"
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
                                target_price=0.0,
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
            try:
                stock = yf.Ticker(symbol)
                news = stock.news
                if news:
                    for article in news:
                        news_id = f"{symbol}_{article.get('id', article.get('guid', ''))}"
                        if news_id not in self.processed_news:
                            self.monitor_stock(symbol)
                            self.processed_news.add(news_id)
            except Exception as e:
                logger.error(f"Error processing news for {symbol}: {str(e)}")
                continue

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
                "confidence_score": 0.0,
                "best_article": None,
                "prediction_stats": {
                    "total": 0,
                    "significant": 0,
                    "aligned": 0,
                    "high_confidence": 0
                }
            }

            # Calculate weighted sentiment scores with time decay
            total_weight = 0
            weighted_sentiment = 0
            weighted_confidence = 0
            
            for article in articles:
                if 'sentiment' in article and article['sentiment']:
                    sentiment = article['sentiment']
                    # Calculate time-based weight (more recent articles have higher weight)
                    pub_time = datetime.fromtimestamp(article['article']['providerPublishTime'], tz=pytz.UTC)
                    hours_old = (datetime.now(tz=pytz.UTC) - pub_time).total_seconds() / 3600
                    weight = 1.0 / (1.0 + hours_old * 0.1)  # Slower decay
                    
                    # Track weighted scores
                    weighted_sentiment += sentiment['score'] * weight
                    weighted_confidence += sentiment['confidence'] * weight
                    total_weight += weight
                    
                    # Track statistics
                    result["prediction_stats"]["total"] += 1
                    if sentiment['confidence'] >= self.signal_criteria['min_confidence']:
                        result["prediction_stats"]["high_confidence"] += 1

            # Calculate final weighted scores
            if total_weight > 0:
                result["sentiment_score"] = weighted_sentiment / total_weight
                result["confidence_score"] = weighted_confidence / total_weight

            # Find best prediction across timeframes and articles
            best_prediction = 0.0
            best_timeframe = None
            best_article = None

            for article in articles:
                if 'predictions' in article:
                    for timeframe, prediction in article['predictions'].items():
                        threshold = self.thresholds.get(timeframe, float('inf'))
                        
                        # Only consider predictions that meet threshold
                        if abs(prediction) >= threshold:
                            result["prediction_stats"]["significant"] += 1
                            
                            # Check sentiment alignment (using continuous score)
                            sentiment_score = article['sentiment']['score']
                            prediction_direction = np.sign(prediction)
                            sentiment_direction = np.sign(sentiment_score)
                            
                            # Consider aligned if directions match and confidence is good
                            if prediction_direction == sentiment_direction and \
                               article['sentiment']['confidence'] >= self.signal_criteria['min_confidence']:
                                result["prediction_stats"]["aligned"] += 1
                            
                            # Update best prediction if current is stronger
                            if abs(prediction) > abs(best_prediction):
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

                # Enhanced buy signal criteria
                threshold_met = abs(best_prediction) >= self.thresholds.get(best_timeframe, float('inf'))
                confidence_met = result["confidence_score"] >= self.signal_criteria['min_confidence']
                sentiment_met = abs(result["sentiment_score"]) >= self.signal_criteria['min_sentiment_score']
                
                # Calculate alignment rate only for significant predictions
                alignment_rate = (result["prediction_stats"]["aligned"] / result["prediction_stats"]["significant"]) \
                               if result["prediction_stats"]["significant"] > 0 else 0
                alignment_met = alignment_rate >= self.signal_criteria['min_alignment_rate']

                # Additional check for neutral probability
                neutral_prob = best_article['sentiment']['probabilities']['neutral']
                neutral_check = neutral_prob <= self.signal_criteria['max_neutral_probability']

                result["should_buy"] = all([
                    threshold_met,
                    confidence_met,
                    sentiment_met,
                    alignment_met,
                    neutral_check
                ])

                # Detailed logging of decision factors
                logger.info(f"\nDecision Factors for {symbol}:")
                logger.info(f"Best Timeframe: {best_timeframe}")
                logger.info(f"Prediction Strength: {best_prediction:.2f}%")
                logger.info(f"Sentiment Score: {result['sentiment_score']:.2f}")
                logger.info(f"Confidence Score: {result['confidence_score']:.2f}")
                logger.info(f"Neutral Probability: {neutral_prob:.2f}")
                logger.info(f"Alignment Rate: {alignment_rate:.2f}")
                logger.info(f"Threshold Met: {threshold_met}")
                logger.info(f"Confidence Met: {confidence_met}")
                logger.info(f"Sentiment Met: {sentiment_met}")
                logger.info(f"Alignment Met: {alignment_met}")
                logger.info(f"Neutral Check: {neutral_check}")
                logger.info(f"Buy Signal Generated: {result['should_buy']}")

            return result

        except Exception as e:
            logger.error(f"Error calculating predictions for {symbol}: {str(e)}")
            return {
                "should_buy": False,
                "best_timeframe": None,
                "prediction_strength": 0.0,
                "target_price": 0.0,
                "sentiment_score": 0.0,
                "confidence_score": 0.0,
                "best_article": None,
                "prediction_stats": {
                    "total": 0,
                    "significant": 0,
                    "aligned": 0,
                    "high_confidence": 0
                }
            }

    async def poll_news(self, symbols: List[str]):
        """Poll news for a list of symbols in batches"""
        try:
            # Store symbols for use in process_market_news
            self.symbols = symbols
            
            # Process symbols in batches of 10
            batch_size = 10
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_number}/{total_batches} ({len(batch)} symbols)")
                
                # Process each symbol in the batch
                tasks = [
                    self.monitor_stock_with_tracking(symbol, batch_number, total_batches)
                    for symbol in batch
                ]
                await asyncio.gather(*tasks)
                
                # Wait 5 seconds between batches to avoid rate limiting
                if i + batch_size < len(symbols):
                    await asyncio.sleep(5)
            
            # Wait 1 minute before next polling cycle
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in poll_news: {str(e)}")
            # Wait 1 minute before retrying on error
            await asyncio.sleep(60)

async def main():
    logger.info("🚀 Market Monitor Starting...")
    logger.info("Initializing ML models and market data...")
    monitor = RealTimeMonitor()
    
    # Get symbols and start monitoring
    symbols = get_all_symbols()
    logger.info(f"Loaded {len(symbols)} symbols")
    await monitor.start(symbols)

if __name__ == "__main__":
    asyncio.run(main())




