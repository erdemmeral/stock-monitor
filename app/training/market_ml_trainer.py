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
import time
import os
import joblib
import logging
from pathlib import Path
import torch
from transformers import BertTokenizer, BertForSequenceClassification
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
        self.vectorizer = TfidfVectorizer()
        self.symbols = self.get_symbols()
        # Modified SGDRegressor initialization with better parameters
        self.models = {
            '1h': SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                tol=1e-3,
                early_stopping=True
            ),
            '1wk': SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                tol=1e-3,
                early_stopping=True
            ),
            '1mo': SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                tol=1e-3,
                early_stopping=True
            )
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
        max_retries = 3  # Maximum number of retry attempts
        retry_count = 0
        while retry_count < max_retries:
            try:
                stock = yf.Ticker(symbol)
                
                # Use cached data
                try:
                    news = stock.news
                except json.JSONDecodeError:
                    logger.error(f"Error getting news for {symbol}: JSON decode error")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying {symbol} (Attempt {retry_count + 1}/{max_retries})")
                        time.sleep(5)  # Wait 5 seconds before retrying
                        continue
                    else:
                        logger.error(f"Failed to get news for {symbol} after {max_retries} attempts")
                        return []
                except Exception as e:
                    logger.error(f"Error getting news for {symbol}: {str(e)}")
                    return []
                    
                symbol_data = []
                processed_articles = 0
                max_articles = 10 
                
                for i, article in enumerate(news, 1):
            # Break if max articles reached
                    if processed_articles >= max_articles:
                        logger.info(f"Reached max articles limit for {symbol}")
                        break
                    
                    try:
                        # Extract article details
                        title = article.get('title', '')
                        link = article.get('link', '')
                        
                        # Log article being processed
                        logger.info(f"Processing article {i}/{len(news)} for {symbol}")
                        logger.debug(f"Article title: {title}")
                        logger.debug(f"Article link: {link}")
                        
                        # Get full article text
                        try:
                            full_text = self.get_full_article_text(link) or ''
                        except Exception as text_error:
                            logger.error(f"Error getting full article text for {symbol}: {text_error}")
                            full_text = ''
                        
                        # Combine content
                        content = title + ' ' + full_text
                        
                        # Skip very short content
                        if len(content.strip()) < 10:
                            logger.warning(f"Skipping article with insufficient content")
                            continue
                        
                        # Analyze sentiment
                        try:
                            sentiment = self.finbert_analyzer.analyze_sentiment(content)
                            
                            # Ensure sentiment is not None
                            if not sentiment:
                                logger.warning(f"Sentiment analysis returned None for an article")
                                continue
                        except Exception as sentiment_error:
                            logger.error(f"Sentiment analysis error for {symbol}: {sentiment_error}")
                            continue
                        
                        # Analyze stock price changes
                        try:
                            publish_date = datetime.fromtimestamp(article['providerPublishTime'])
                            changes = self.analyze_stock(stock, publish_date)
                        except Exception as changes_error:
                            logger.error(f"Error analyzing stock changes for {symbol}: {changes_error}")
                            continue
                        
                        # Create sample if changes exist
                        if changes:
                            sample = {
                                'text': content,
                                'changes': changes,
                                'symbol': symbol,
                                'date': publish_date,
                                'sentiment': sentiment
                            }
                            symbol_data.append(sample)
                            processed_articles += 1
                        
                        # Log progress for each article
                        logger.info(f"Processed article {i}: Sentiment={sentiment['score']:.2f}, Changes={bool(changes)}")
                    
                    except Exception as article_error:
                        logger.error(f"Unexpected error processing article {i} for {symbol}: {article_error}")
                        continue
                
                # Final logging
                logger.info(f"Completed processing {symbol}")
                logger.info(f"Total valid samples: {len(symbol_data)}")
                logger.info(f"Total processed articles: {processed_articles}")
                
                return symbol_data
            
            except Exception as final_error:
                logger.error(f"Critical error in process_symbol for {symbol}: {final_error}")
                logger.exception("Full error traceback")
                return []
    def analyze_stock(self, stock, publish_date):
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning, 
                          message='The \'unit\' keyword in TimedeltaIndex construction is deprecated')
        changes = {}
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
                changes[timeframe] = change
                if not pd.isna(change):  # Only add valid changes
                    changes[timeframe] = change
                    if abs(change) > 1.0:  # Only log significant changes
                        logger.info(f"{stock.ticker}: {timeframe} change of {change:.2f}%")
                    
            except Exception as e:
                logger.error(f" Error processing {stock.ticker} for {timeframe}: {str(e)}")
                    
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
            
            # Get sentiment features
            sentiment = sample.get('sentiment', {})
            if not sentiment:
                return None, None
            
            # Calculate average impact across timeframes
            avg_impact = np.mean(list(impact_scores.values()))
            
            # Adjust sentiment features based on historical impact
            sentiment_features = [
                sentiment['score'] * (1 + avg_impact),  # Amplify sentiment if historically impactful
                sentiment['confidence'],
                sentiment['probabilities']['positive'] * (1 + avg_impact),
                sentiment['probabilities']['negative'] * (1 + avg_impact),
                sentiment['probabilities']['neutral']
            ]
            
            # Convert to sparse matrix
            sentiment_sparse = scipy.sparse.csr_matrix(sentiment_features).reshape(1, -1)
            
            # Combine features
            combined_features = scipy.sparse.hstack([sample_tfidf, sentiment_sparse])
            
            return combined_features, avg_impact
            
        except Exception as e:
            logger.error(f"Error preparing training sample: {str(e)}")
            return None, None

    def train_with_impact_scores(self, training_data, timeframe):
        """Train model with impact-adjusted samples"""
        try:
            X_features_list = []
            y = []
            sample_weights = []
            
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
                
                X_features_list.append(features)
                y.append(sample['changes'].get(timeframe, 0))
                
                # Use impact as sample weight
                weight = 1.0 + impact_scores.get(timeframe, 0)
                sample_weights.append(weight)
            
            if not X_features_list:
                return False
                
            # Combine features
            X_combined = scipy.sparse.vstack(X_features_list)
            y_array = np.array(y, dtype=float)
            weights_array = np.array(sample_weights)
            
            # Remove NaN values
            valid_mask = ~np.isnan(y_array)
            X_clean = X_combined[valid_mask]
            y_clean = y_array[valid_mask]
            weights_clean = weights_array[valid_mask]
            
            if len(y_clean) == 0:
                return False
            
            # Train model with sample weights
            self.models[timeframe].fit(X_clean, y_clean, sample_weight=weights_clean)
            
            # Log training summary
            logger.info(f"\nTraining Summary for {timeframe}:")
            logger.info(f"Total samples: {len(y_clean)}")
            logger.info(f"Average sample weight: {np.mean(weights_clean):.2f}")
            logger.info(f"Max sample weight: {np.max(weights_clean):.2f}")
            
            # Show sample predictions
            sample_indices = np.random.choice(len(y_clean), min(3, len(y_clean)), replace=False)
            logger.info("\nSample predictions vs actual:")
            for idx in sample_indices:
                pred = self.models[timeframe].predict(X_clean[idx:idx+1])[0]
                actual = y_clean[idx]
                weight = weights_clean[idx]
                logger.info(f"Pred: {pred:.1f}% | Actual: {actual:.1f}% | Weight: {weight:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in train_with_impact_scores: {str(e)}")
            return False

    def collect_and_train(self, symbols):
        logger.info(" Starting training process...")
        training_data = []
        failed_stocks = set()
        sentiment_impact_analysis = {
                '1h': {
                    'total_samples': 0,
                    'sentiment_distribution': {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    },
                    'price_changes_by_sentiment': {
                        'positive': [],
                        'negative': [],
                        'neutral': []
                    }
                },
                '1wk': {
                    'total_samples': 0,
                    'sentiment_distribution': {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    },
                    'price_changes_by_sentiment': {
                        'positive': [],
                        'negative': [],
                        'neutral': []
                    }
                },
                '1mo': {
                    'total_samples': 0,
                    'sentiment_distribution': {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    },
                    'price_changes_by_sentiment': {
                        'positive': [],
                        'negative': [],
                        'neutral': []
                    }
                }
            }





         # Configure yfinance cache
        cache_dir = Path("./cache/yfinance")
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        # Determine number of processes (use 75% of available cores)
        num_processes = 4  # Can increase processes
         # Process symbols in smaller chunks
        chunk_size = 50  # Process 50 symbols at a time
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            logger.info(f"\nProcessing chunk {i//chunk_size + 1}/{len(symbols)//chunk_size + 1}")
            logger.info(f"Chunk symbols: {chunk}")

            with Pool(processes=num_processes) as pool:
                results = []
                for symbol, result in zip(chunk, pool.imap_unordered(self.process_symbol, chunk)):
                    if result and len(result) > 0:  # Check if we got actual data
                        results.extend(result)
                        logger.info(f"Successfully processed {symbol} with {len(result)} samples")

                    else:  # If no data was returned
                        failed_stocks.add(symbol)
                        logger.info(f"Failed to process {symbol}")

                training_data.extend(results)
                logger.info(f"Total samples collected so far: {len(training_data)}")
                logger.info(f"Failed stocks: {failed_stocks}")

            
            # Add delay between chunks
            logger.info("Taking a break between chunks...")
            time.sleep(10)  # 10 second break between chunks


        if failed_stocks:
            logger.info(f"\nRetrying {len(failed_stocks)} failed stocks")
            retry_chunk_size = 25  # Smaller chunk size for retries
            
            for retry_attempt in range(2):  # Try up to 2 more times
                if not failed_stocks:
                    break
                    
                logger.info(f"Retry attempt {retry_attempt + 1}/2 for {len(failed_stocks)} stocks")
                retry_symbols = list(failed_stocks)
                still_failed = set()
                
                # Process retry chunks with Pool
                for i in range(0, len(retry_symbols), retry_chunk_size):
                    retry_chunk = retry_symbols[i:i+retry_chunk_size]
                    logger.info(f"\nProcessing retry chunk {i//retry_chunk_size + 1}/{len(retry_symbols)//retry_chunk_size + 1}")
                    
                    with Pool(processes=num_processes) as pool:
                        retry_results = []
                        for symbol, result in zip(retry_chunk, pool.imap_unordered(self.process_symbol, retry_chunk)):
                            if result and len(result) > 0:
                                retry_results.extend(result)
                                failed_stocks.remove(symbol)
                                logger.info(f"Successfully processed {symbol} on retry attempt {retry_attempt + 1}")
                            else:
                                still_failed.add(symbol)
                                logger.info(f"Failed to process {symbol} on retry attempt {retry_attempt + 1}")
                        
                        training_data.extend(retry_results)
                    
                    # Add longer delay between retry chunks
                    logger.info("Taking a break between retry chunks...")
                    time.sleep(15)  # 15 second break between retry chunks
                
                failed_stocks = still_failed
                if failed_stocks:
                    logger.info(f"Still failed after retry attempt {retry_attempt + 1}: {len(failed_stocks)} stocks")
                    logger.info("Waiting 30 seconds before next retry attempt...")
                    time.sleep(30)  # Longer wait between retry attempts
            
            if failed_stocks:
                logger.info("\nPermanently failed stocks:")
                for symbol in sorted(failed_stocks):
                    logger.info(f"- {symbol}")
        
        logger.info("\nTraining Data Summary:")
        logger.info(f"Total samples collected: {len(training_data)}")
        logger.info(f"Final failed stocks count: {len(failed_stocks)}")
        
        # Train models if we have data
        if training_data:
            logger.info("\nPreparing to train models...")
            logger.info(f"Total training samples collected: {len(training_data)}")
            
            # Fit the vectorizer BEFORE training
            X_text = [sample['text'] for sample in training_data]
            X = self.vectorizer.fit_transform(X_text)
            logger.info(f"Text features shape: {X.shape}")
            
            # Prepare features for each timeframe
            for timeframe in ['1h', '1wk', '1mo']:
                logger.info(f"\nTraining {timeframe} model...")
                
                # Use sparse matrix and memory-efficient processing
                from scipy.sparse import hstack
                
                # Prepare features and target values
                y = []
                X_features_list = []
                sample_weights = []
                
                for sample in training_data:
                    try:
                        # Calculate impact score based on sentiment prediction accuracy
                        sentiment_direction = np.sign(sample['sentiment']['score'])
                        price_change = sample['changes'].get(timeframe, 0)
                        price_direction = np.sign(price_change)
                        
                        # Calculate base impact score
                        impact_score = abs(price_change) / self.thresholds[timeframe]
                        
                        # Adjust impact based on prediction accuracy
                        if sentiment_direction == price_direction:
                            impact_score *= 1.5  # Boost weight for correct predictions
                        else:
                            impact_score *= 0.5  # Reduce weight for incorrect predictions
                            
                        # Add time decay factor (more recent articles have higher weight)
                        days_old = (datetime.now(tz=pytz.UTC) - sample['date']).days
                        time_decay = 1.0 / (1.0 + days_old * 0.1)  # Decay factor
                        
                        # Final sample weight
                        sample_weight = (1.0 + impact_score) * time_decay
                        
                        # Get TF-IDF features
                        sample_tfidf = self.vectorizer.transform([sample['text']])
                        
                        # Add sentiment features
                        if sample.get('sentiment'):
                            sentiment = sample['sentiment']
                            # Adjust sentiment features based on historical accuracy
                            sentiment_features = [
                                sentiment['score'] * (1 + impact_score),  # Amplify if historically accurate
                                sentiment['confidence'],
                                sentiment['probabilities']['positive'],
                                sentiment['probabilities']['negative'],
                                sentiment['probabilities']['neutral']
                            ]
                            sentiment_sparse = scipy.sparse.csr_matrix(sentiment_features).reshape(1, -1)
                            combined_features = scipy.sparse.hstack([sample_tfidf, sentiment_sparse])
                            X_features_list.append(combined_features)
                        else:
                            X_features_list.append(sample_tfidf)
                        
                        y.append(price_change)
                        sample_weights.append(sample_weight)
                        
                    except Exception as e:
                        logger.error(f"Error processing sample: {str(e)}")
                        continue
                
                # Combine features and convert to arrays
                try:
                    X_combined = scipy.sparse.vstack(X_features_list)
                    y_array = np.array(y, dtype=float)
                    weights_array = np.array(sample_weights)
                    
                    # Handle NaN values
                    valid_mask = ~np.isnan(y_array)
                    X_clean = X_combined[valid_mask]
                    y_clean = y_array[valid_mask]
                    weights_clean = weights_array[valid_mask]
                    
                    if len(y_clean) > 0:
                        logger.info(f"Training {timeframe} model with {len(y_clean)} samples")
                        logger.info(f"Average sample weight: {np.mean(weights_clean):.2f}")
                        
                        # Train model with sample weights
                        self.models[timeframe].fit(X_clean, y_clean, sample_weight=weights_clean)
                        
                        # Show sample predictions with weights
                        sample_indices = np.random.choice(len(y_clean), min(3, len(y_clean)), replace=False)
                        logger.info("\nSample predictions vs actual:")
                        for idx in sample_indices:
                            pred = self.models[timeframe].predict(X_clean[idx:idx+1])[0]
                            actual = y_clean[idx]
                            weight = weights_clean[idx]
                            logger.info(f"Pred: {pred:.1f}% | Actual: {actual:.1f}% | Weight: {weight:.2f}")
                    
                except Exception as array_error:
                    logger.error(f"Error preparing training data for {timeframe}: {array_error}")
                    continue

            logger.info("\nTraining Complete!")
            logger.info(f"Total symbols processed: {len(symbols)}")
            logger.info(f"Total training samples: {len(training_data)}")
                
            return self.models
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
        Make a prediction with sentiment integration
        
        :param text: Input text
        :param timeframe: Prediction timeframe
        :return: Predicted price change
        """
        # Vectorize text
        X_tfidf = self.vectorizer.transform([text])
        
        # Analyze sentiment
        sentiment = self.finbert_analyzer.analyze_sentiment(text)
        
        # Get base prediction
        base_pred = self.models[timeframe].predict(X_tfidf)[0]
        
        # Adjust prediction with sentiment
        if sentiment:
            sentiment_multiplier = self._get_sentiment_multiplier(sentiment)
            adjusted_pred = base_pred * sentiment_multiplier
            
            # Log detailed prediction breakdown
            logger.info(f"Prediction Breakdown for {timeframe}:")
            logger.info(f"Base Prediction: {base_pred:.2f}%")
            logger.info(f"Sentiment: {sentiment['score']:.2f}")
            logger.info(f"Sentiment Multiplier: {sentiment_multiplier:.2f}")
            logger.info(f"Adjusted Prediction: {adjusted_pred:.2f}%")
            
            return adjusted_pred
    
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
    trainer = MarketMLTrainer()
    trainer.collect_and_train(trainer.symbols)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(" Starting Market ML Training...")
    trainer = MarketMLTrainer()
    symbols = trainer.get_symbols()
    logger.info(f"Loaded {len(symbols)} symbols")
    
    try:
        models = trainer.collect_and_train(symbols)
        if models:
            logger.info(" Training completed successfully!")
            trainer.save_models()  # This line is properly in the Python code
            logger.info(" Models saved!")

        else:
            logger.error(" Training failed!")
    except Exception as e:
        logger.error(f" Training error: {str(e)}")
