from urllib.parse import urlparse
import warnings
from bs4 import BeautifulSoup
import numpy as np
import scipy.sparse


import requests
import telegram
from telegram.error import NetworkError, Unauthorized

from app.models.trade_history import TradeHistory
warnings.filterwarnings('ignore', message='.*The \'unit\' keyword in TimedeltaIndex.*')

# Standard library imports
import asyncio
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, List
import random
import os
import logging
# Third-party imports
from aiohttp import web
import aiohttp
import yfinance as yf
import joblib
import pandas as pd  # Add this import
import pytz         # Also add this if not already there


# Local application imports
from app.models.stock_analyzer import StockAnalyzer
from app.services.news_service import NewsService
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.utils.sp500 import get_all_symbols
from app.models.stock import StockAnalysis
from app.models.news import NewsArticle
from app.utils.telegram_notifier import TelegramNotifier
from app.models.position import Position
import torch
from app.models.portfolio import Portfolio
from app.training.market_ml_trainer import FinBERTSentimentAnalyzer
from app.services.portfolio_tracker_service import PortfolioTrackerService

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
        self.portfolio = Portfolio()
        self.trade_history = TradeHistory()
        
        # Initialize portfolio tracker with proper logging
        logger.info("Initializing PortfolioTrackerService")
        self.portfolio_tracker = PortfolioTrackerService()
        
        self.stop_loss_percentage = 5.0
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        self.sentiment_weight = 0.3  # Configurable weight for sentiment impact

        self.model_manager = ModelManager()
        
        # Use models from model manager
        self.vectorizer = self.model_manager.models['vectorizer']
        self.models = {
            '1h': self.model_manager.models['1h'],
            '1wk': self.model_manager.models['1wk'],
            '1mo': self.model_manager.models['1mo']
        }
        for timeframe, model in self.models.items():
            logger.info(f"Model loaded for {timeframe}: {type(model)}")
        for timeframe in ['1h', '1wk', '1mo']:
            model_path = f'app/models/market_model_{timeframe}.joblib'
            creation_time = datetime.fromtimestamp(os.path.getctime(model_path))
            logger.info(f"Model {timeframe} created: {creation_time}")

            logger.info(f"Vectorizer created: {datetime.fromtimestamp(os.path.getctime('app/models/vectorizer.joblib'))}")    
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        # Initialize Telegram bot
        try:
            if not hasattr(self, 'telegram_bot'):
                self.telegram_bot = telegram.Bot(token=self.telegram_token)
                logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
                
        self.news_service = NewsService()
        self.sentiment_analyzer = SentimentAnalyzer()  # Add this line
        self.processed_news = set()
        self.portfolio = Portfolio()

        self.thresholds = {
            '1h': 5.0,
            '1wk': 10.0,
            '1mo': 20.0
        }


        self._polling_lock = asyncio.Lock()  # Add this line
        self._is_polling = False  # Add this line

    def _pad_features(self, X):
        """
        Pad or truncate features to match expected dimensions
        """
        try:
            # Expected number of features during training
            expected_features = 95721
            
            # Ensure X is a sparse matrix
            if not scipy.sparse.issparse(X):
                X = scipy.sparse.csr_matrix(X)
            
            # Current number of features
            current_features = X.shape[1]
            
            if current_features == expected_features:
                return X
            
            # If fewer features, pad with zeros
            if current_features < expected_features:
                padding_size = expected_features - current_features
                padding = scipy.sparse.csr_matrix(
                    np.zeros((X.shape[0], padding_size))
                )
                return scipy.sparse.hstack([X, padding])
            
            # If more features, truncate
            return X[:, :expected_features]
        
        except Exception as e:
            logger.error(f"Feature padding error: {e}")
            logger.error(f"Input matrix shape: {X.shape}")
            return X
    def predict_with_sentiment(self, text, timeframe):
        """
        Make a prediction integrating FinBERT sentiment
        """
        try:
            # Vectorize text
            X_tfidf = self.vectorizer.transform([text])
            
            # Base prediction from model
            base_pred = self.models[timeframe].predict(X_tfidf)[0]
            
            # Analyze sentiment
            sentiment = self.finbert_analyzer.analyze_sentiment(text)
            
            # If sentiment available, adjust prediction
            if sentiment:
                # Sentiment multiplier logic
                multiplier_map = {
                    'negative': 0.7,   # Reduce prediction
                    'neutral': 1.0,    # No change
                    'positive': 1.3    # Increase prediction
                }
                
                # Confidence-based adjustment
                confidence = max(sentiment['probabilities'].values())
                base_multiplier = multiplier_map.get(sentiment['label'], 1.0)
                
                # Adjusted prediction
                adjusted_pred = base_pred * (
                    1 + (base_multiplier - 1) * self.sentiment_weight * confidence
                )
                
                # Logging for transparency
                logger.info(f"Prediction for {timeframe}:")
                logger.info(f"Base Prediction: {base_pred:.2f}%")
                logger.info(f"Sentiment: {sentiment['label']}")
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

    
    async def analyze_and_notify(self, symbol: str, articles: list, prices: pd.DataFrame):
    # Initialize deadlines
        logger.info(f"📊 Comprehensive Stock Analysis: {symbol}")
        logger.info(f"Total News Articles Analyzed: {len(articles)}")

        deadlines = {
            '1h': timedelta(hours=1),
            '1wk': timedelta(days=7),
            '1mo': timedelta(days=30)
        }
        analysis_summary = {
                'symbol': symbol,
                'total_articles': len(articles),
                'valid_predictions': {},
                'decision': 'No Action'
        }
        # Start message with stock alert and articles summary
        message = f"🔔 <b>Stock Alert: {symbol}</b>\n\n"
        message += f"📰 <b>Recent News Articles:</b>\n"
        
        # Add all articles with their dates
        for article_data in articles:
            article = article_data['article']
            publish_time = datetime.fromtimestamp(article['providerPublishTime'], tz=pytz.UTC)
            message += f"• {article['title']} ({publish_time.strftime('%Y-%m-%d')})\n"

        # Calculate valid timeframes and aggregate predictions
        current_time = datetime.now(tz=pytz.UTC)
        aggregated_preds = {}
        valid_timeframes = {}
        for timeframe in self.thresholds.keys():
            valid_preds = []
            timeframe_analysis = {
                'total_predictions': 0,
                'valid_predictions': [],
                'mean_prediction': 0,
                'meets_threshold': False
            }
            for article_data in articles:
                predictions = article_data['predictions']
                publish_time = datetime.fromtimestamp(article_data['article']['providerPublishTime'], tz=pytz.UTC)
                time_passed = current_time - publish_time
                deadline = deadlines[timeframe]

                # Check if at least 70% of the timeframe is still available
                time_remaining_ratio = (deadline - time_passed) / deadline

                if (timeframe in predictions and 
                time_passed < deadline and 
                time_remaining_ratio >= 0.7):
                
                    valid_preds.append(predictions[timeframe])
                    timeframe_analysis['valid_predictions'].append(predictions[timeframe])


            if valid_preds:
                aggregated_pred = sum(valid_preds) / len(valid_preds)
                threshold = self.thresholds[timeframe]

                timeframe_analysis.update({
                    'total_predictions': len(valid_preds),
                    'mean_prediction': aggregated_pred,
                    'meets_threshold': aggregated_pred > 0 and aggregated_pred >= threshold
                })
                #Log detailed timeframe analysis
                logger.info(f"🔍 {timeframe} Analysis:")
                logger.info(f"  Total Predictions: {timeframe_analysis['total_predictions']}")
                logger.info(f"  Mean Prediction: {aggregated_pred:.2f}%")
                logger.info(f"  Threshold: {threshold}%")
                logger.info(f"  Meets Threshold: {timeframe_analysis['meets_threshold']}")

                # Only include if prediction is positive AND exceeds threshold
                if aggregated_pred > 0 and aggregated_pred >= threshold:
                    valid_timeframes[timeframe] = aggregated_pred
                    analysis_summary['valid_predictions'][timeframe] = {
                        'prediction': aggregated_pred,
                        'score': aggregated_pred / threshold
                    }

        if not valid_timeframes:
            return  # Don't send message if no predictions meet criteria
        # Add predictions section
        message += f"\n⏱ <b>Valid Predictions Remaining:</b>\n"
        
        for timeframe, pred in valid_timeframes.items():
            threshold = self.thresholds[timeframe]
            score = pred / threshold
            latest_publish = max([datetime.fromtimestamp(art['article']['providerPublishTime'], tz=pytz.UTC) 
                                for art in articles])
            deadline = latest_publish + deadlines[timeframe]
            
            message += (
                f"• {timeframe}: {pred:.2f}% "
                f"(Score: {score:.2f}x threshold of {threshold}%)\n"
                f"  Based on {len(articles)} news articles\n"
                f"  Deadline: {deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            )

        # Find and add strongest signal
        if valid_timeframes:
            best_timeframe = max(valid_timeframes.items(), 
                            key=lambda x: abs(x[1]/self.thresholds[x[0]]))
            if abs(best_timeframe[1]/self.thresholds[best_timeframe[0]]) >= 1.0:
                current_price = prices['Close'].iloc[-1]
                timeframe = best_timeframe[0]
                expected_change = best_timeframe[1]
                target_price = current_price * (1 + expected_change/100)
                
                # Define target dates
                target_dates = {
                    '1h': timedelta(hours=1),
                    '1wk': timedelta(days=7),
                    '1mo': timedelta(days=30)
                }
                
                # Calculate target date
                current_time = datetime.now(tz=pytz.UTC)
                target_date = current_time + target_dates[timeframe]

                # Ensure all required arguments are passed
                if not self.portfolio.has_position(symbol):
                    self.portfolio.add_position(
                    symbol=symbol,
                    entry_price=current_price,
                    target_price=target_price,
                    target_date=target_date,
                    timeframe=timeframe,
                    current_price=current_price,  # Explicitly pass current price
                    entry_date=current_time  # Explicitly pass entry date
                    )
                    logger.info(f"Added new position: {symbol} at ${current_price:.2f}")
                    logger.info(f"Target Price: ${target_price:.2f}")
                    logger.info(f"Target Date: {target_date}")
                analysis_summary['decision'] = f"BUY - {best_timeframe[0]} Signal"
            
                # Log decision details
                logger.info("🟢 BUY SIGNAL DETECTED")
                logger.info(f"Best Timeframe: {best_timeframe[0]}")
                logger.info(f"Prediction Strength: {best_timeframe[1]:.2f}%")
                
               
                # Add to portfolio if first cycle is complete
                
                


                timeframe_text = {'1h': '1 hour', '1wk': '1 week', '1mo': '1 month'}[best_timeframe[0]]
                prediction = best_timeframe[1]
                score = prediction / self.thresholds[best_timeframe[0]]
                
                message += (
                    f"\n🎯 <b>Strongest Valid Signal:</b>\n"
                    f"Timeframe: {timeframe_text}\n"
                    f"Expected Change: {prediction:.2f}%\n"
                    f"Score: {score:.2f}x threshold\n"
                    f"Based on {len(articles)} news articles\n"
                    f"Latest Deadline: {deadline.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
        else:
            logger.info("🔘 No Strong Signals - No Action")
        # Check if predictions already happened

        # Log final analysis summary
        logger.info("📈 Stock Analysis Summary:")
        logger.info(f"Symbol: {analysis_summary['symbol']}")
        logger.info(f"Total Articles: {analysis_summary['total_articles']}")
        logger.info(f"Valid Predictions: {list(analysis_summary['valid_predictions'].keys())}")
        logger.info(f"Final Decision: {analysis_summary['decision']}")

        opportunity_messages = []
        is_missed_opportunity = False  # Flag to track if this is a missed opportunity

        for timeframe, pred in valid_timeframes.items():
            if pred <= 0:
                continue
            earliest_publish = min([datetime.fromtimestamp(art['article']['providerPublishTime'], tz=pytz.UTC) 
                                for art in articles])
            mask = prices.index >= earliest_publish
            period_prices = prices[mask]
            
            if not period_prices.empty:
                start_price = period_prices.iloc[0]['Close']
                highest_price = period_prices['High'].max()
                max_change = ((highest_price - start_price) / start_price) * 100
                
                # Only check for upward movements
                if max_change >= pred:
                    is_missed_opportunity = True

                    max_date = period_prices[period_prices['High'] == highest_price].index[0]
                    days_taken = (max_date - earliest_publish).days
                    opportunity_messages.append(
                        f"\n⚠️ <b>Missed Opportunity:</b>\n"
                        f"The predicted +{pred:.2f}% movement for {timeframe} already occurred!\n"
                        f"Price moved +{max_change:.2f}% within {days_taken} days after the first news.\n"
                        f"Max price reached on: {max_date.strftime('%Y-%m-%d')}"
                    )
    # Only add to portfolio if:
        # 1. First cycle is complete
        # 2. It's not a missed opportunity
        # 3. We have valid signals
        if not is_missed_opportunity and valid_timeframes:
            best_timeframe = max(valid_timeframes.items(), 
                            key=lambda x: abs(x[1]/self.thresholds[x[0]]))
            if abs(best_timeframe[1]/self.thresholds[best_timeframe[0]]) >= 1.0:
                current_price = prices['Close'].iloc[-1]
                timeframe = best_timeframe[0]
                expected_change = best_timeframe[1]
                target_price = current_price * (1 + expected_change/100)
                best_article = max(
                    [
                        article for article in articles 
                        if timeframe in article['predictions'] and 
                        article['predictions'][timeframe] > 0
                    ],
                    key=lambda x: abs(x['predictions'][timeframe]),
                    default=None
                )
                target_dates = {
                    '1h': timedelta(hours=1),
                    '1wk': timedelta(days=7),
                    '1mo': timedelta(days=30)
                }

                if best_article:
                    publish_time = datetime.fromtimestamp(best_article['article']['providerPublishTime'], tz=pytz.UTC)
                    target_date = publish_time + target_dates[timeframe]
                    logger.info(f"Target date based on news article published at {publish_time}")
                else:
                    current_time = datetime.now(tz=pytz.UTC)
                    target_date = current_time + target_dates[timeframe]
                    logger.info("No specific news article found, using current time for target date")
                # Add to portfolio only if we're not already tracking this stock
                if not self.portfolio.has_position(symbol):
                    self.portfolio.add_position(
                        symbol=symbol,
                        entry_price=current_price,
                        target_price=target_price,
                        target_date=target_date,
                        timeframe=timeframe,
                        current_price=current_price,
                        entry_date=current_time
                    )

                    logger.info(f"Added new position: {symbol} at ${current_price:.2f}")
                    logger.info(f"Target Price: ${target_price:.2f}")
                    logger.info(f"Target Date: {target_date}")
                else:
                    logger.info(f"Already tracking position for {symbol}")

        # Add opportunity messages to the alert if any
        if opportunity_messages:
            message += "\n" + "\n".join(opportunity_messages)

        asyncio.create_task(self.send_telegram_alert(message))

        return analysis_summary
       
    async def handle_telegram_update(self, request):
        try:
            data = await request.json()
            logger.info(f"Received telegram update: {data}")  # Log the incoming update
            
            if 'message' in data and 'text' in data['message']:
                command = data['message']['text']
                chat_id = data['message']['chat']['id']
                logger.info(f"Received command: {command} from chat_id: {chat_id}")

                if command == '/start' or command == '/help':
                    await self.send_help_message(chat_id)
                elif command == '/portfolio':
                    await self.send_portfolio_status(chat_id)
                elif command == '/history':
                    await self.send_trading_history(chat_id)

        except Exception as e:
            logger.error(f"Error handling telegram update: {e}")
            logger.exception(e)  # This will print the full traceback
        
        return web.Response(text='OK')

    async def send_help_message(self, chat_id):
        try:
            bot = self.telegram_bot.bot  # Get bot instance
            logger.info(f"Sending help message to chat_id: {chat_id}")
            help_message = (
                "🤖 <b>Stock Monitor Bot Commands</b>\n\n"
                
                "📝 <b>Available Commands:</b>\n\n"
                
                "/help - Show this help message\n"
                "Get list of all available commands and their descriptions\n\n"
                
                "/portfolio - View current positions\n"
                "See all active positions with entry prices, current P/L, and target prices\n\n"
                
                "/history - View trading history\n"
                "See recent trades, total P/L, and overall performance metrics\n\n"
                
                "ℹ️ <b>About Alerts:</b>\n"
                "• Buy signals are sent automatically when significant opportunities are detected\n"
                "• Sell signals are sent when:\n"
                "  - Target price is reached 🎯\n"
                "  - Stop loss is triggered ⚠️\n"
                "  - Target date is reached 📅\n\n"
                
                "📊 <b>Performance Tracking:</b>\n"
                "• All trades are automatically recorded\n"
                "• Use /history to view performance metrics\n"
                "• Use /portfolio to track current positions"
            )

            self.telegram_bot.send_message(
                chat_id=chat_id,
                text=help_message,
                parse_mode='HTML'
            )
            logger.info("Help message sent successfully")
        except Exception as e:
            logger.error(f"Error sending help message: {str(e)}")
            logger.exception(e)
    async def send_trading_history(self, chat_id):
            

            if not self.trade_history.trades:
                message = "📊 <b>Trading History</b>\n\nNo completed trades yet."
            else:
                message = "📊 <b>Trading History</b>\n\n"
                
                # Last 5 trades
                message += "<b>Recent Trades:</b>\n"
                for trade in self.trade_history.trades[-5:]:
                    message += (
                        f"🔄 {trade['symbol']}: {trade['profit_loss_percentage']:+.2f}%\n"
                        f"   ${trade['profit_loss']:+.2f}\n"
                        f"   {trade['reason']}\n\n"
                    )

                # Summary statistics
                total_pl = self.trade_history.get_total_profit_loss()
                win_rate = self.trade_history.get_win_rate()
                
                message += (
                    f"📈 <b>Overall Performance:</b>\n"
                    f"Total P/L: ${total_pl:+.2f}\n"
                    f"Win Rate: {win_rate:.1f}%\n"
                    f"Total Trades: {len(self.trade_history.trades)}"
                )

            self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML'
            )
    async def send_portfolio_status(self, chat_id):
        if not self.portfolio.positions:
            message = "📊 <b>Portfolio Status</b>\n\nNo active positions."
        else:
            message = "📊 <b>Portfolio Status</b>\n\n"
            total_pl_percent = 0
            total_pl_dollar = 0

            for symbol, position in self.portfolio.positions.items():
                pl_dollar = position.current_price - position.entry_price
                pl_percent = (pl_dollar / position.entry_price) * 100
                total_pl_dollar += pl_dollar
                total_pl_percent += pl_percent

                # Calculate position duration
                duration = position.get_position_duration()
                
                # Format entry date
                entry_date_formatted = position.entry_date.strftime('%Y-%m-%d %H:%M:%S UTC')

                message += (
                    f"🎯 <b>{symbol}</b>\n"
                    f"Started: {entry_date_formatted}\n"
                    f"Entry Price: ${position.entry_price:.2f}\n"
                    f"Current Price: ${position.current_price:.2f}\n"
                    f"Target: ${position.target_price:.2f}\n"
                    f"P/L: ${pl_dollar:.2f} ({pl_percent:+.2f}%)\n"
                    f"Held for: {duration['days']} days, {duration['hours']} hours\n"
                    f"Target date: {position.target_date.strftime('%Y-%m-%d')} "
                    f"({(position.target_date - datetime.now(tz=pytz.UTC)).days} days remaining)\n"
                    f"Timeframe: {position.timeframe}\n\n"
                )

            # Add portfolio summary
            num_positions = len(self.portfolio.positions)
            avg_pl_percent = total_pl_percent / num_positions if num_positions > 0 else 0
            
            message += (
                f"📈 <b>Portfolio Summary</b>\n"
                f"Active Positions: {num_positions}\n"
                f"Total P/L: ${total_pl_dollar:.2f}\n"
                f"Average P/L: {avg_pl_percent:.2f}%"
            )

        try:
            self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error sending portfolio status: {e}")

    async def send_signal_to_localhost(self, signal_data: dict):
        """
        Send signal to local host
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post('http://localhost:5000/api/signals', json=signal_data) as response:
                    if response.status == 200:
                        logger.info(f"Signal sent to localhost successfully: {signal_data}")
                    else:
                        logger.error(f"Failed to send signal to localhost. Status: {response.status}")
        except Exception as e:
            logger.error(f"Error sending signal to localhost: {e}")

    async def send_buy_signal(self, symbol: str, sentiment_score: float, price_data: Dict) -> None:
        """Send buy signal to Telegram and portfolio tracker"""
        try:
            # 1. Calculate prices and targets
            current_price = price_data['current_price']
            price_targets = self.calculate_price_targets(current_price, price_data['price_movements'])
            trading_targets, trailing_stop = self.calculate_trading_targets(price_data['price_movements'])
            
            # 2. Send Telegram alert first
            buy_message = (
                f"🔔 <b>BUY Signal Generated!</b>\n\n"
                f"Symbol: {symbol}\n"
                f"Entry Price: ${current_price:.2f}\n"
                f"Target Price: ${price_targets['1mo']:.2f}\n"
                f"Sentiment Score: {sentiment_score:.2f}\n\n"
                f"Expected Movement:\n"
                f"• 1 Hour: {trading_targets['short_term']:.2f}%\n"
                f"• 1 Week: {trading_targets['mid_term']:.2f}%\n"
                f"• 1 Month: {trading_targets['long_term']:.2f}%\n\n"
                f"🛡 Trailing Stop: {trailing_stop}%"
            )
            await self.send_telegram_alert(buy_message)
            logger.info(f"✅ Telegram alert sent for {symbol}")
            
            # 3. Then send to portfolio tracker
            entry_date = datetime.now(tz=pytz.UTC)
            target_date = entry_date + timedelta(days=30)
            
            if not hasattr(self, 'portfolio_tracker') or self.portfolio_tracker is None:
                logger.info("Initializing portfolio tracker service")
                self.portfolio_tracker = PortfolioTrackerService()
            
            # Send buy signal with explicit float conversion and proper date formatting
            success = await self.portfolio_tracker.send_buy_signal(
                symbol=symbol,
                entry_price=float(current_price),
                target_price=float(price_targets['1mo']),
                entry_date=entry_date.isoformat(),
                target_date=target_date.isoformat()
            )
            
            if success:
                logger.info(f"✅ Buy signal successfully sent to portfolio tracker for {symbol}")
            else:
                logger.error(f"❌ Failed to send buy signal to portfolio tracker for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in send_buy_signal for {symbol}: {str(e)}", exc_info=True)
            raise

    def calculate_price_targets(self, current_price: float, price_movements: Dict) -> Dict:
        """Calculate price targets based on predicted movements"""
        targets = {}
        for timeframe, movement in price_movements.items():
            target = current_price * (1 + movement/100)
            targets[timeframe] = target
        return targets

    def calculate_trading_targets(self, price_movements: Dict) -> tuple:
        """Calculate trading targets and trailing stop"""
        trading_targets = {
            'short_term': price_movements.get('1h', 0),
            'mid_term': price_movements.get('1wk', 0),
            'long_term': price_movements.get('1mo', 0)
        }
        
        # Calculate trailing stop based on volatility
        movements = list(price_movements.values())
        volatility = np.std(movements) if movements else 0
        trailing_stop = max(5.0, min(volatility * 2, 15.0))  # Between 5% and 15%
        
        return trading_targets, trailing_stop

    async def send_telegram_alert(self, message: str) -> None:
        """Send alert message via Telegram"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            raise

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
        try:
        # Send startup message to Telegram
            startup_message = (
                "🚀 <b>Stock Monitor Activated</b>\n\n"
                f"📊 Monitoring {len(symbols)} stocks\n"
                f"🕒 Started at: {datetime.now(tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                "🔍 Initializing market monitoring process..."
            )
        
            # Send startup alert
            await self.send_telegram_alert(startup_message)
        except Exception as alert_error:
            logger.error(f"Failed to send startup Telegram alert: {alert_error}")

        logger.info(f"Starting monitoring for {len(symbols)} symbols...")
        
        
        
        """Start monitoring"""
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

            while True:  # Add continuous loop
                try:
                    logger.info("Starting new monitoring cycle...")
                    
                            # Process stocks in batches for better resource management
                    batch_size = 50  # Adjust based on your system's capabilities
                    total_symbols = len(symbols)
                    
                    for batch_start in range(0, total_symbols, batch_size):
                        batch = symbols[batch_start:batch_start+batch_size]
                        batch_number = batch_start // batch_size + 1
                        total_batches = (total_symbols + batch_size - 1) // batch_size
                        
                        logger.info(f"Processing Batch {batch_number}/{total_batches} "
                                    f"(Stocks {batch_start+1}-{min(batch_start+batch_size, total_symbols)})")
                        
                        # Create tasks with tracking
                        # Use more aggressive concurrency
                        async def process_symbols():
                            # Create semaphore to limit concurrent tasks
                            sem = asyncio.Semaphore(50)  # Limit to 50 concurrent tasks
                            
                            async def bounded_monitor(symbol):
                                async with sem:
                                    return await self.monitor_stock_with_tracking(symbol, batch_number, total_batches)
                            
                            # Create tasks with the semaphore
                            tasks = [bounded_monitor(symbol) for symbol in batch]
                            return await asyncio.gather(*tasks)
                
                        # Process batch concurrently
                        await process_symbols()

                        logger.info(f"Completed Batch {batch_number}/{total_batches}")
                
                # Small delay between batches to prevent overwhelming resources
                        await asyncio.sleep(1)
                    
                    
                    
                    logger.info("Completed full monitoring cycle. Waiting 5 minutes before next cycle...")
                    await asyncio.sleep(300)  # 5-minute break between cycles
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {str(e)}")
                    await asyncio.sleep(300)  # 5-minute break on error
        finally:
            self._is_polling = False
            await polling_task

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
            one_week_ago = datetime.now(tz=pytz.UTC) - timedelta(days=7)

            logger.info(f"🔄 Processing stock: {symbol}")
            stock = yf.Ticker(symbol)
            
            try:
                prices = stock.history(period="3mo", interval="1h")
                if prices.empty:
                    logger.warning(f"Skipping {symbol}: No recent price data available")
                    return
            except Exception as e:
                logger.warning(f"Skipping {symbol}: {str(e)}")
                return

            # Filter news when fetching
            news = [
                article for article in stock.news 
                if datetime.fromtimestamp(article['providerPublishTime'], tz=pytz.UTC) >= one_week_ago
            ]

            if not news:
                logger.info(f"ℹ️ No recent news (last 7 days) found for {symbol}")
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

                # Process one article completely before moving to next
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
                       # Vectorize text
                    X = self.vectorizer.transform([content])
                
                    # Pad features BEFORE prediction
                    X_padded = self._pad_features(X)
                    
                    for timeframe, model in self.models.items():
                        try:
                        # Base prediction
                            base_pred = model.predict(X_padded)[0]
                            
                            # Adjust prediction with sentiment if available
                            if sentiment:
                                # Sentiment multiplier
                                multiplier_map = {
                                    'negative': 0.7,   # Reduce prediction
                                    'neutral': 1.0,    # No change
                                    'positive': 1.3    # Increase prediction
                                }
                                
                                # Confidence-based adjustment
                                confidence = max(sentiment['probabilities'].values())
                                base_multiplier = multiplier_map.get(sentiment['label'], 1.0)
                                
                                # Adjusted prediction
                                pred = base_pred * (1 + (base_multiplier - 1) * 0.3 * confidence)
                            else:
                                pred = base_pred
                            
                            article_predictions[timeframe] = pred
                            logger.info(f"{timeframe} prediction: {pred:.2f}")
                        
                        except Exception as e:
                            # Log detailed error information
                            logger.error(f"Error in prediction for {timeframe}: {e}")
                            logger.error(f"Input matrix shape: {X_padded.shape}")
                            logger.error(f"Model expected features: {model.coef_.shape[1]}")
                            continue
                    
                    # Store predictions
                    all_predictions.append({
                        'article': article,
                        'predictions': article_predictions,
                        'sentiment': sentiment
                    })
                    
                    self.processed_news.add(news_id)
                    processed_urls.add(url)

                    # Add small delay before next article
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing article for {symbol}: {str(e)}")
                    continue
            
            # Analyze and notify if predictions exist
            if all_predictions:
                await self.analyze_and_notify(
                    symbol=symbol,
                    articles=all_predictions,
                    prices=prices
                )
        
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {str(e)}")
            return

    async def update_positions(self):
        logger.info("Updating portfolio positions...")
        current_time = datetime.now(tz=pytz.UTC)
        positions_to_remove = []

        for symbol, position in self.portfolio.positions.items():
            try:
                stock = yf.Ticker(symbol)
                current_price = float(stock.info.get('regularMarketPrice', 0))
                
                if current_price <= 0:
                    continue

                position.current_price = current_price
                price_change_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                 # Check if target price achieved
                if current_price >= position.target_price:
                    await self.send_sell_signal(
                        symbol, 
                        reason=f"🎯 Target price ${position.target_price:.2f} achieved! ({price_change_percent:.2f}% gain)",
                        position=position
                    )
                    positions_to_remove.append(symbol)
                    continue

                # Check for stop loss
                if price_change_percent <= -self.stop_loss_percentage:
                    await self.send_sell_signal(
                        symbol, 
                        reason=f"Stop loss triggered: Price dropped {abs(price_change_percent):.2f}%",
                        position=position
                    )
                    positions_to_remove.append(symbol)
                    continue

                # Check if target date has passed
                if current_time >= position.target_date:
                    if price_change_percent > 0:
                        reason = f"Target date reached with {price_change_percent:.2f}% profit"
                    else:
                        reason = f"Target date reached with {price_change_percent:.2f}% loss"
                    await self.send_sell_signal(symbol, reason=reason, position=position)
                    positions_to_remove.append(symbol)

            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {str(e)}")

        # Remove closed positions
        for symbol in positions_to_remove:
            self.portfolio.remove_position(symbol)
        logger.info(f"Portfolio update complete. Active positions: {len(self.portfolio.positions)}")

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

    async def send_telegram_alert(self, message):
        try:
            logger.info("Attempting to send Telegram alert...")
            logger.info(f"Message content:\n{message}")
            
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            logger.info("✅ Telegram alert sent successfully")
        except Exception as e:
            logger.error(f"❌ Telegram notification error: {str(e)}")
            logger.error(f"Failed message content:\n{message}")
         # Add a small delay after sending
        await asyncio.sleep(1)

    def analyze_price_movement(self, prices, publish_time) -> List[Dict]:
        publish_date = datetime.fromtimestamp(publish_time).date()
        future_prices = prices[prices.index.date > publish_date]
        
        movements = []
        check_periods = [
            ('1h', 1/24),
            ('1d', 1),
            ('1w', 7),
            ('1m', 30),
            ('3m', 90)
        ]
        
        for label, days in check_periods:
            end_date = publish_date + timedelta(days=days)
            period_prices = future_prices[future_prices.index.date <= end_date]
            
            if len(period_prices) > 0:
                start_price = period_prices['Close'].iloc[0]
                end_price = period_prices['Close'].iloc[-1]
                change_pct = ((end_price - start_price) / start_price) * 100
                
                movements.append({
                    'period': label,
                    'change_pct': round(change_pct, 2)
                })
        
        return movements

    def update_portfolio_sentiment(self, symbol: str, new_sentiment: float) -> None:
        if symbol in self.portfolio:
            current = self.portfolio[symbol]
            current['sentiment_score'] = (current['sentiment_score'] + new_sentiment) / 2
            
            if current['sentiment_score'] <= self.sentiment_threshold_sell:
                asyncio.create_task(self.send_sell_signal(symbol))

    def should_ignore_news(self, headline: str, text: str = "") -> bool:
        """
        Filter out news that could give false signals
        """
        ignore_phrases = [
            'reverse stock split',
            'reverse split',
            'consolidation of shares',
            'share consolidation',
            'consolidates shares',
            'consolidating shares'
        ]
        
        # Combine headline and text for checking
        full_text = f"{headline} {text}".lower()
        
        # Check for ignore phrases
        for phrase in ignore_phrases:
            if phrase in full_text:
                logger.info(f"Ignoring news containing '{phrase}': {headline}")
                return True
                
        return False
    async def handle_health_check(self, request):
        return web.Response(text="Bot is running!")
    async def poll_telegram_updates(self):
        """Poll for Telegram updates"""
        async with self._polling_lock:  # Use lock to ensure single instance
            try:
                # Clear existing updates first using synchronous call
                result = self.telegram_bot.getUpdates(offset=-1, timeout=1)
                if result:
                    offset = result[-1].update_id + 1
                else:
                    offset = None
                logger.info("Cleared existing updates")
            except Exception as e:
                logger.error(f"Error clearing updates: {str(e)}")
                offset = None

            while self._is_polling:
                try:
                    # Get updates using synchronous call with proper timeout
                    updates = self.telegram_bot.getUpdates(
                        offset=offset,
                        timeout=10,  # Reduced timeout
                        allowed_updates=['message'],
                        limit=100  # Limit number of updates per request
                    )
                    
                    if updates:
                        for update in updates:
                            if update.message and update.message.text:
                                command = update.message.text
                                chat_id = update.message.chat.id
                                logger.info(f"Received command: {command} from chat_id: {chat_id}")

                                try:
                                    # Handle commands
                                    if command == '/start' or command == '/help':
                                        await self.send_help_message(chat_id)
                                    elif command == '/portfolio':
                                        await self.send_portfolio_status(chat_id)
                                    elif command == '/history':
                                        await self.send_trading_history(chat_id)
                                except Exception as cmd_error:
                                    logger.error(f"Error handling command {command}: {str(cmd_error)}")

                                # Update offset after processing each update
                                offset = update.update_id + 1
                                logger.debug(f"Updated offset to {offset}")

                    # Small delay between polling requests
                    await asyncio.sleep(1)
                
                except Exception as e:
                    logger.error(f"Error polling telegram updates: {str(e)}")
                    await asyncio.sleep(5)  # Longer delay on error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

# Set more verbose logging for monitoring
logging.getLogger('app.services.news_service').setLevel(logging.INFO)
logging.getLogger('app.services.stock_analyzer').setLevel(logging.INFO)
logging.getLogger('app.services.sentiment_analyzer').setLevel(logging.INFO)

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




