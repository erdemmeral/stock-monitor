import json
from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone
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
import traceback
from pathlib import Path
import torch
from dtaidistance import dtw
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.base import clone
import xgboost as xgb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from requests import Session
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import psutil
import gc
from sklearn.base import BaseEstimator, ClassifierMixin

# Create logs directory if it doesn't exist
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a unique log file name with timestamp
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Configure logging to write to both file and console with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will print to console
    ]
)

# Set logging level for yfinance to WARNING to reduce noise
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure our logger is set to INFO level

# Log the start of the session with clear separator
logger.info("="*80)
logger.info(f"Starting new training session. Logging to: {log_filename}")
logger.info("="*80)

# Then replace the print statements with logger calls
class NewsSemanticAnalyzer:
    def __init__(self, embedding_model=None):
        # Use shared embedding model if provided
        self.embedding_model = embedding_model
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # We don't need separate BERT models since we're using SentenceTransformer
        self.model = None
        self.tokenizer = None

        # Create data storage directories
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.embeddings_dir = os.path.join(self.data_dir, 'embeddings')
        self.clusters_dir = os.path.join(self.data_dir, 'clusters')
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.clusters_dir, exist_ok=True)

        # In-memory index with size limits
        self.embedding_index = {}  # Maps ID to file location
        self.cluster_index = {}    # Maps cluster ID to file location
        self.max_embeddings = 10000  # Maximum number of embeddings to keep in memory
        self.max_clusters = 1000   # Maximum number of clusters to keep in memory
        
        # Batch processing settings
        self.batch_size = 100
        self.current_batch = {
            'embeddings': [],
            'timestamps': [],
            'impacts': []
        }
        
        self.cluster_data = {
            '1wk': {
                'centroids': None,
                'embeddings': [],
                'timestamps': [],
                'price_changes': [],
                'cluster_labels': None,
                'cluster_stats': {}
            },
            '1mo': {
                'centroids': None,
                'embeddings': [],
                'timestamps': [],
                'price_changes': [],
                'cluster_labels': None,
                'cluster_stats': {}
            }
        }
        
        # Add cleanup settings
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
        self.data_ttl = 86400 * 7    # 7 days

    def cleanup_memory(self):
        """Clean up memory periodically"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        try:
            # Cleanup embeddings
            if len(self.embedding_index) > self.max_embeddings:
                # Keep only the most recent embeddings
                sorted_embeddings = sorted(self.embedding_index.items(), key=lambda x: x[1]['timestamp'])
                self.embedding_index = dict(sorted_embeddings[-self.max_embeddings:])

            # Cleanup clusters
            if len(self.cluster_index) > self.max_clusters:
                sorted_clusters = sorted(self.cluster_index.items(), key=lambda x: x[1]['timestamp'])
                self.cluster_index = dict(sorted_clusters[-self.max_clusters:])

            # Cleanup cluster data
            for timeframe in self.cluster_data:
                data = self.cluster_data[timeframe]
                if len(data['embeddings']) > self.max_embeddings:
                    # Keep only the most recent data
                    data['embeddings'] = data['embeddings'][-self.max_embeddings:]
                    data['timestamps'] = data['timestamps'][-self.max_embeddings:]
                    data['price_changes'] = data['price_changes'][-self.max_embeddings:]
                    if data['cluster_labels'] is not None:
                        data['cluster_labels'] = data['cluster_labels'][-self.max_embeddings:]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.last_cleanup = current_time
            logger.info("Memory cleanup completed")

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")

    def cleanup_old_files(self):
        """Clean up old cache files"""
        try:
            current_time = time.time()
            
            # Clean up embeddings directory
            for filename in os.listdir(self.embeddings_dir):
                file_path = os.path.join(self.embeddings_dir, filename)
                if current_time - os.path.getmtime(file_path) > self.data_ttl:
                    os.remove(file_path)
                    logger.info(f"Removed old embedding file: {filename}")
            
            # Clean up clusters directory
            for filename in os.listdir(self.clusters_dir):
                file_path = os.path.join(self.clusters_dir, filename)
                if current_time - os.path.getmtime(file_path) > self.data_ttl:
                    os.remove(file_path)
                    logger.info(f"Removed old cluster file: {filename}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")

    def get_embedding(self, text):
        """Generate embedding for text using BERT"""
        try:
            self.cleanup_memory()  # Periodic cleanup
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
        finally:
            gc.collect()

    def cluster_news(self, embeddings, timestamps=None, price_changes=None, timeframe=None, eps=0.25):
        """Enhanced clustering with temporal patterns and saving cluster information"""
        try:
            if len(embeddings) < 2:
                logger.warning("Not enough embeddings for clustering")
                return [-1] * len(embeddings)

            # Convert to numpy array
            embeddings = np.array(embeddings)
            
            # Timestamps check
            if timestamps is None:
                timestamps = [datetime.now()] * len(embeddings)
            
            # Prepare time features with more weight on temporal proximity
            max_timestamp = max(timestamps)
            time_features = np.array([(max_timestamp - t).total_seconds() / 86400 for t in timestamps])
            time_features = (time_features - np.min(time_features)) / (np.max(time_features) - np.min(time_features) + 1e-6)
            time_features = time_features.reshape(-1, 1)
            
            # Normalize embeddings
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Combine features with more weight on semantic similarity
            combined_features = np.hstack([embeddings_norm * 0.8, time_features * 0.2])
            
            # DBSCAN clustering
            clustering = DBSCAN(
                eps=eps,
                min_samples=3,
                metric='cosine',
                n_jobs=-1
            ).fit(combined_features)
            
            labels = clustering.labels_
            
            # Store cluster information if timeframe is provided
            if timeframe and timeframe in self.cluster_data:
                self.cluster_data[timeframe]['embeddings'] = embeddings
                self.cluster_data[timeframe]['timestamps'] = timestamps
                self.cluster_data[timeframe]['price_changes'] = price_changes
                self.cluster_data[timeframe]['cluster_labels'] = labels
                
                # Calculate and store cluster centroids and stats
                unique_labels = set(labels)
                cluster_stats = {}
                centroids = {}
                
                for label in unique_labels:
                    if label != -1:
                        mask = labels == label
                        cluster_embeddings = embeddings[mask]
                        cluster_times = np.array(timestamps)[mask]
                        cluster_prices = np.array(price_changes)[mask]
                        
                        # Calculate centroid
                        centroid = np.mean(cluster_embeddings, axis=0)
                        centroids[label] = centroid
                        
                        # Calculate statistics
                        cluster_stats[label] = {
                            'size': len(cluster_embeddings),
                            'avg_price_change': float(np.mean(cluster_prices)),
                            'std_price_change': float(np.std(cluster_prices)),
                            'time_span': (max(cluster_times) - min(cluster_times)).total_seconds() / 3600,
                            'avg_similarity': float(np.mean([
                                1 - cosine(e1, e2)
                                for i, e1 in enumerate(cluster_embeddings)
                                for e2 in cluster_embeddings[i+1:]
                            ] if len(cluster_embeddings) > 1 else [1.0]))
                        }
                
                self.cluster_data[timeframe]['centroids'] = centroids
                self.cluster_data[timeframe]['cluster_stats'] = cluster_stats
                
                # Save cluster data
                self.save_cluster_data(timeframe)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in cluster_news: {str(e)}")
            return [-1] * len(embeddings)
    
    def _extract_temporal_patterns(self, ordered_embeddings, window_size=3):
        """Extract temporal patterns using Euclidean distance and correlation"""
        patterns = []
        
        # Sliding window over embeddings
        for i in range(len(ordered_embeddings) - window_size + 1):
            window = ordered_embeddings[i:i + window_size]
            patterns.append(window)
        
        if not patterns:
            return []
        
        # Cluster similar patterns
        pattern_similarities = np.zeros((len(patterns), len(patterns)))
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Calculate both Euclidean distance and correlation
                eucl_dist = euclidean(patterns[i].flatten(), patterns[j].flatten())
                try:
                    corr, _ = pearsonr(patterns[i].flatten(), patterns[j].flatten())
                    corr_dist = 1 - abs(corr)  # Convert correlation to distance
                except:
                    corr_dist = 1.0  # Maximum distance if correlation fails
                
                # Combine distances (weighted average)
                combined_dist = 0.7 * eucl_dist + 0.3 * corr_dist
                pattern_similarities[i, j] = combined_dist
                pattern_similarities[j, i] = combined_dist
        
        # Cluster patterns
        pattern_clusters = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='precomputed'
        ).fit(pattern_similarities)
        
        # Extract representative patterns
        unique_patterns = []
        for cluster_id in np.unique(pattern_clusters.labels_):
            if cluster_id == -1:
                continue
            
            # Get patterns in this cluster
            cluster_patterns = [patterns[i] for i in range(len(patterns)) 
                              if pattern_clusters.labels_[i] == cluster_id]
            
            # Use centroid as representative pattern
            centroid = np.mean(cluster_patterns, axis=0)
            unique_patterns.append(centroid)
        
        return unique_patterns

    def analyze_cluster_impact(self, cluster_id, timeframe):
        """Enhanced cluster impact analysis with temporal weighting"""
        if cluster_id not in self.cluster_impacts:
            return None

        impacts = self.cluster_impacts[cluster_id]
        if not impacts:
            return None

        # Get timestamps for this cluster
        cluster_times = [self.news_timestamps[idx] for idx in self.news_clusters[cluster_id]]
        
        # Calculate time-based weights
        now = datetime.now()
        time_weights = []
        
        for t in cluster_times:
            age_days = (now - t).total_seconds() / 86400.0  # Convert to days
            
            # Dynamic decay based on timeframe
            if timeframe == '1h':
                decay_factor = 2.0  # Faster decay for short-term
            elif timeframe == '1wk':
                decay_factor = 1.0  # Medium decay
            else:  # 1mo
                decay_factor = 0.5  # Slower decay for long-term
            
            weight = np.exp(-decay_factor * age_days)
            time_weights.append(weight)
        
        # Combine with impact values
        impacts_array = np.array([impact[timeframe] for impact in impacts if timeframe in impact])
        if len(impacts_array) == 0:
            return None
            
        # Normalize weights
        time_weights = np.array(time_weights)
        time_weights = time_weights / np.sum(time_weights)
        
        # Calculate weighted statistics
        weighted_mean = np.average(impacts_array, weights=time_weights)
        weighted_std = np.sqrt(np.average((impacts_array - weighted_mean) ** 2, weights=time_weights))
        
        # Analyze temporal patterns if available
        temporal_confidence = 0.0
        if cluster_id in self.temporal_patterns:
            patterns = self.temporal_patterns[cluster_id]
            if patterns:
                # Calculate pattern consistency
                pattern_similarities = []
                for pattern in patterns:
                    # Compare with recent data
                    recent_data = impacts_array[-len(pattern):]
                    if len(recent_data) == len(pattern):
                        # Use both Euclidean distance and correlation
                        eucl_dist = euclidean(recent_data, pattern)
                        try:
                            corr, _ = pearsonr(recent_data, pattern)
                            similarity = (1.0 / (1.0 + eucl_dist)) * (0.5 + 0.5 * abs(corr))
                        except:
                            similarity = 1.0 / (1.0 + eucl_dist)
                        pattern_similarities.append(similarity)
                
                if pattern_similarities:
                    temporal_confidence = np.mean(pattern_similarities)

        # Calculate final confidence score
        base_confidence = 1.0 - (weighted_std / (abs(weighted_mean) + 1e-6))
        temporal_weight = 0.3  # Weight for temporal pattern confidence
        final_confidence = (1 - temporal_weight) * base_confidence + temporal_weight * temporal_confidence

        return {
            'mean': np.mean(impacts_array),
            'weighted_mean': weighted_mean,
            'std': weighted_std,
            'median': np.median(impacts_array),
            'count': len(impacts_array),
            'temporal_confidence': temporal_confidence,
            'final_confidence': final_confidence,
            'time_decay_factor': decay_factor,
            'recent_weight': np.max(time_weights)
        }

    def find_similar_news(self, embedding, threshold=0.8):
        """Enhanced similar news finding with temporal pattern matching"""
        if not self.news_embeddings:
            return []

        # Calculate base similarities
        similarities = cosine_similarity([embedding], self.news_embeddings)[0]
        
        # Get temporal patterns for recent news
        recent_window = 50  # Look at last 50 news items
        if len(self.news_embeddings) > recent_window:
            recent_embeddings = np.array(self.news_embeddings[-recent_window:])
            recent_patterns = self._extract_temporal_patterns(recent_embeddings)
        else:
            recent_patterns = []
        
        # Calculate temporal pattern similarity
        pattern_bonus = np.zeros_like(similarities)
        if recent_patterns:
            for idx in range(len(self.news_embeddings)):
                # Look for pattern matches
                if idx >= self.WINDOW_SIZE - 1:
                    current_window = self.news_embeddings[idx-self.WINDOW_SIZE+1:idx+1]
                    
                    # Compare with known patterns
                    pattern_similarities = []
                    for pattern in recent_patterns:
                        if len(current_window) == len(pattern):
                            similarity = 1.0 / (1.0 + dtw.distance(current_window, pattern))
                            pattern_similarities.append(similarity)
                    
                    if pattern_similarities:
                        pattern_bonus[idx] = np.max(pattern_similarities) * 0.2  # 20% bonus for pattern match

        # Combine similarities with pattern bonus
        final_similarities = similarities + pattern_bonus
        
        # Find similar articles with enhanced matching
        similar_indices = []
        for idx, sim in enumerate(final_similarities):
            if sim >= threshold:
                similar_indices.append((idx, sim))
        
        return sorted(similar_indices, key=lambda x: x[1], reverse=True)[:5]

    def update_patterns(self, text, price_impacts):
        """Update patterns with new data"""
        try:
            embedding = self.get_embedding(text)
            if embedding is None:
                logger.warning("Failed to generate embedding for text")
                return

            # Store new data
            self.news_embeddings.append(embedding)
            self.price_impacts.append(price_impacts)
            self.timestamps.append(time.time())
            
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

    def get_semantic_impact_prediction(self, embedding, timeframe):
        """Predict price impact based on semantic similarity to historical patterns"""
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

    def save_cluster_data(self, timeframe):
        """Save cluster data for future predictions"""
        try:
            cluster_file = os.path.join('app', 'models', f'cluster_data_{timeframe}.joblib')
            joblib.dump(self.cluster_data[timeframe], cluster_file)
            logger.info(f"Saved cluster data for {timeframe}")
            return True
        except Exception as e:
            logger.error(f"Error saving cluster data: {str(e)}")
            return False
    
    def load_cluster_data(self, timeframe):
        """Load saved cluster data"""
        try:
            cluster_file = os.path.join('app', 'models', f'cluster_data_{timeframe}.joblib')
            self.cluster_data[timeframe] = joblib.load(cluster_file)
            logger.info(f"Loaded cluster data for {timeframe}")
            return True
        except Exception as e:
            logger.error(f"Error loading cluster data: {str(e)}")
            return False

    def predict_cluster(self, text, timeframe, max_distance=0.3):
        """Predict cluster membership for a new article"""
        try:
            if not self.cluster_data[timeframe]['centroids']:
                if not self.load_cluster_data(timeframe):
                    return None
            
            # Get embedding for new text
            embedding = self.get_embedding(text)
            if embedding is None:
                return None
            
            # Normalize embedding
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            # Find closest cluster
            best_cluster = None
            min_distance = float('inf')
            
            for label, centroid in self.cluster_data[timeframe]['centroids'].items():
                distance = cosine(embedding_norm, centroid)
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_cluster = label
            
            if best_cluster is not None:
                cluster_info = {
                    'cluster_id': best_cluster,
                    'similarity_score': 1 - min_distance,
                    'cluster_stats': self.cluster_data[timeframe]['cluster_stats'][best_cluster],
                    'prediction_confidence': 1 - (min_distance / max_distance)
                }
                return cluster_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error predicting cluster: {str(e)}")
            return None

    def cleanup_old_data(self):
        """Cleanup old data to prevent memory growth"""
        try:
            if len(self.news_embeddings) > self.max_stored_items:
                # Keep only the most recent items
                self.news_embeddings = self.news_embeddings[-self.max_stored_items:]
                self.price_impacts = self.price_impacts[-self.max_stored_items:]
                self.timestamps = self.timestamps[-self.max_stored_items:]
                self.news_timestamps = self.news_timestamps[-self.max_stored_items:]

            # Cleanup clusters
            for cluster_id in list(self.news_clusters.keys()):
                if len(self.news_clusters[cluster_id]) > self.max_stored_items:
                    self.news_clusters[cluster_id] = self.news_clusters[cluster_id][-self.max_stored_items:]
                    self.cluster_impacts[cluster_id] = self.cluster_impacts[cluster_id][-self.max_stored_items:]

            # Cleanup temporal patterns
            for pattern_id in list(self.temporal_patterns.keys()):
                if len(self.temporal_patterns[pattern_id]) > self.max_stored_items:
                    self.temporal_patterns[pattern_id] = self.temporal_patterns[pattern_id][-self.max_stored_items:]

            # Cleanup cluster data
            for timeframe in self.cluster_data:
                if len(self.cluster_data[timeframe]['embeddings']) > self.max_stored_items:
                    self.cluster_data[timeframe]['embeddings'] = self.cluster_data[timeframe]['embeddings'][-self.max_stored_items:]
                    self.cluster_data[timeframe]['timestamps'] = self.cluster_data[timeframe]['timestamps'][-self.max_stored_items:]
                    self.cluster_data[timeframe]['price_changes'] = self.cluster_data[timeframe]['price_changes'][-self.max_stored_items:]

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Successfully cleaned up old data")
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}")

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

class NewsDataset(Dataset):
    def __init__(self, embeddings, targets, max_seq_length=10):
        self.embeddings = embeddings
        self.targets = targets
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        embedding_seq = self.embeddings[idx]
        target = self.targets[idx]
        
        # Pad sequence if needed
        if len(embedding_seq) < self.max_seq_length:
            padding = torch.zeros(
                (self.max_seq_length - len(embedding_seq), embedding_seq.shape[1])
            )
            embedding_seq = torch.cat([embedding_seq, padding], dim=0)
        else:
            embedding_seq = embedding_seq[:self.max_seq_length]
        
        return embedding_seq, target, len(embedding_seq)

class HybridMarketPredictor(BaseEstimator, ClassifierMixin):
    """Model that learns historical relationships between news content and price changes"""
    
    def __init__(self):
        self.n_features_in_ = None
        self.linear_weights = None
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.lstm_model = None  # Will be initialized after getting feature size
        
    def fit(self, X, y, semantic_features=None):
        """Learn historical relationships between news features and price changes"""
        try:
            # Store number of features
            self.n_features_in_ = X.shape[1]
            
            # Convert sparse matrix to dense if needed
            if scipy.sparse.issparse(X):
                X = X.toarray()
            
            # Add semantic features if provided
            if semantic_features is not None:
                X = np.hstack([X, semantic_features])
                self.n_features_in_ = X.shape[1]
            
            # Convert y to numpy array if needed
            y = np.array(y)
            
            # Remove extreme price changes
            mask = np.abs(y) <= 50
            if not np.any(mask):
                raise ValueError("No valid price changes within reasonable range")
            
            X = X[mask]
            y = y[mask]
            
            logger.info("\nTraining Analysis:")
            
            # 1. Linear Regression Analysis
            logger.info("\n1. Linear Impact Analysis:")
            try:
                X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                self.linear_weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
                linear_impact = X_with_bias @ self.linear_weights
                linear_mae = np.mean(np.abs(linear_impact - y))
                logger.info(f"   Average Impact Error: {linear_mae:.4f}%")
                logger.info(f"   Max Positive Impact: {np.max(linear_impact):.2f}%")
                logger.info(f"   Max Negative Impact: {np.min(linear_impact):.2f}%")
            except Exception as e:
                logger.error(f"Error in linear analysis: {str(e)}")
            
            # 2. XGBoost Analysis
            logger.info("\n2. Non-linear Impact Analysis:")
            try:
                self.xgb_model.fit(X, y)
                xgb_impact = self.xgb_model.predict(X)
                xgb_mae = np.mean(np.abs(xgb_impact - y))
                logger.info(f"   Average Impact Error: {xgb_mae:.4f}%")
                
                # Feature importance analysis
                importances = self.xgb_model.feature_importances_
                logger.info(f"   Most influential feature weight: {np.max(importances):.4f}")
                logger.info(f"   Average feature influence: {np.mean(importances):.4f}")
            except Exception as e:
                logger.error(f"Error in XGBoost analysis: {str(e)}")
            
            # 3. LSTM Analysis
            logger.info("\n3. Sequential Impact Analysis:")
            try:
                # Initialize LSTM with correct input size
                self.lstm_model = NewsLSTM(input_size=self.n_features_in_)
                
                # Prepare data for LSTM
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(y)
                dataset = NewsDataset([X_tensor], [y_tensor])
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Train LSTM
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(self.lstm_model.parameters())
                
                for epoch in range(10):
                    total_loss = 0
                    for batch_x, batch_y, _ in dataloader:
                        optimizer.zero_grad()
                        out, _ = self.lstm_model(batch_x)
                        loss = criterion(out.squeeze(), batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 2 == 0:
                        avg_loss = total_loss/len(dataloader)
                        logger.info(f"   Epoch {epoch}: Avg Impact Error = {avg_loss:.4f}%")
                
                # Final impact analysis
                self.lstm_model.eval()
                with torch.no_grad():
                    lstm_impact, _ = self.lstm_model(X_tensor)
                    lstm_mae = np.mean(np.abs(lstm_impact.numpy().squeeze() - y))
                    logger.info(f"   Final Average Impact Error: {lstm_mae:.4f}%")
            except Exception as e:
                logger.error(f"Error in LSTM analysis: {str(e)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

class MarketMLTrainer:
    # Class constants
    TIMEFRAMES = {
        '1wk': {'interval': '1d', 'days': 7, 'min_points': 3},
        '1mo': {'interval': '1d', 'days': 30, 'min_points': 15}
    }

    def __init__(self):
        """Initialize the trainer with required components"""
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.training_data_dir = os.path.join(self.data_dir, 'training')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Initialize shared embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.models = {}
        self.vectorizers = {}
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
        # Pass the shared embedding model
        self.semantic_analyzer = NewsSemanticAnalyzer(embedding_model=self.embedding_model)
        
        # Initialize cluster data with size limits
        self.cluster_data = defaultdict(lambda: {
            'centroids': None,
            'embeddings': [],
            'timestamps': [],
            'price_changes': [],
            'cluster_labels': None,
            'cluster_stats': {},
            'max_size': 10000  # Maximum number of items to keep
        })

        # CPU and memory management
        total_cpus = multiprocessing.cpu_count()
        self.max_workers = max(1, min(total_cpus - 1, 8))
        
        logger.info(f"System has {total_cpus} CPUs, using {self.max_workers} for processing")
        
        # Monitor system resources
        self.process = psutil.Process(os.getpid())
        
        # Memory thresholds (in percentage)
        self.memory_warning_threshold = 75.0
        self.memory_critical_threshold = 85.0
        
        # Processing settings with optimized delays
        self.batch_size = 50
        self.api_chunk_size = 25
        self.delay_between_batches = 0.5
        self.delay_between_chunks = 5
        self.delay_between_symbols = 0.1
        
        # Cache settings
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache management
        self.max_cache_age = 86400  # 24 hours
        self.max_cache_size = 1000
        self.cache_cleanup_threshold = 0.9
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour

        # Add training history tracking
        self.training_history = {
            '1wk': {
                'best_score': float('-inf'),
                'samples_processed': 0,
                'last_improvement': time.time()
            },
            '1mo': {
                'best_score': float('-inf'),
                'samples_processed': 0,
                'last_improvement': time.time()
            }
        }
        
        # Training data backup
        self.training_data_backup = {'1wk': [], '1mo': []}
        self.max_backup_size = 10000  # Maximum samples to keep in backup
        
        # Minimum improvement threshold
        self.min_improvement = 0.01  # 1% improvement required to save model

    def _cleanup_cache(self, force=False):
        """Clean up old cache files"""
        try:
            current_time = time.time()
            if not force and current_time - self.last_cache_cleanup < self.cleanup_interval:
                return

            # Get list of cache files
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_cache.json')]
            
            # Remove old cache files
            for cache_file in cache_files:
                file_path = os.path.join(self.cache_dir, cache_file)
                if current_time - os.path.getmtime(file_path) > self.max_cache_age:
                    os.remove(file_path)
                    logger.info(f"Removed old cache file: {cache_file}")

            # If we're still over the limit, remove oldest files
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_cache.json')]
            if len(cache_files) > self.max_cache_size:
                cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.cache_dir, x)))
                files_to_remove = cache_files[:-self.max_cache_size]
                for file_to_remove in files_to_remove:
                    os.remove(os.path.join(self.cache_dir, file_to_remove))
                    logger.info(f"Removed excess cache file: {file_to_remove}")

            self.last_cache_cleanup = current_time
            gc.collect()

        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")

    def process_symbol(self, symbol):
        """Process a single symbol with optimized disk caching"""
        try:
            self._cleanup_cache()  # Periodic cleanup
            
            cache_file = os.path.join(self.cache_dir, f'{symbol}_results_cache.json')
            
            # Check disk cache
            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 86400:  # Cache valid for 24 hours
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            
            # If not cached, process the symbol
            stock = yf.Ticker(symbol)
            news_items = self.get_news_for_symbol(symbol)
            
            if not news_items:
                logger.warning(f"No news found for {symbol}")
                return []
            
            # Process articles in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_article, article, symbol, stock)
                    for article in news_items
                ]
                samples = [
                    result for future in as_completed(futures)
                    if (result := future.result()) is not None
                ]
            
            # Write final results to cache
            if samples:
                with open(cache_file, 'w') as f:
                    json.dump(samples, f)
            
            # Clear objects
            del stock
            del news_items
            gc.collect()
            
            return samples
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {str(e)}")
            return []
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(self):
        """Main training function"""
        try:
            # 1. Get training data
            symbols = self.get_symbols()
            if not symbols:
                raise Exception("No symbols available for training")

            # 2. Train models for each timeframe
            training_results = self.collect_and_train(symbols)
            if not training_results:
                raise Exception("Training failed")

            # 3. Verify and save models
            for timeframe, success in training_results.items():
                if not success:
                    raise Exception(f"Training failed for {timeframe} model")
                
            self.save_models()
            return True

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False

    def load_models(self):
        """Load all models and components"""
        try:
            logger.info("Loading models and components...")
            
            # Load models for each timeframe
            for timeframe in ['1wk', '1mo']:
                try:
                    model_path = os.path.join(self.models_dir, f'market_model_{timeframe}.joblib')
                    vectorizer_path = os.path.join(self.models_dir, f'vectorizer_{timeframe}.joblib')
                    
                    # Load if files exist
                    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                        self.models[timeframe] = joblib.load(model_path)
                        self.vectorizers[timeframe] = joblib.load(vectorizer_path)
                        logger.info(f"Successfully loaded model components for {timeframe}")
                    else:
                        logger.warning(f"Model files not found for {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Error loading model components for {timeframe}: {str(e)}")
                    logger.exception("Full traceback:")
                    continue
            
            logger.info("Model loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def save_models(self):
        """Save all models and components"""
        try:
            logger.info("Saving models and components...")
            
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Save models for each timeframe
            for timeframe in self.models.keys():
                try:
                    # Save model and vectorizer
                    model_path = os.path.join(self.models_dir, f'market_model_{timeframe}.joblib')
                    vectorizer_path = os.path.join(self.models_dir, f'vectorizer_{timeframe}.joblib')
                    
                    # Save components
                    joblib.dump(self.models[timeframe], model_path)
                    joblib.dump(self.vectorizers[timeframe], vectorizer_path)
                    
                    logger.info(f"Successfully saved model components for {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error saving model components for {timeframe}: {str(e)}")
                    logger.exception("Full traceback:")
                    continue
            
            logger.info("All models and components saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def evaluate_model(self, timeframe, validation_data):
        """Evaluate model performance on validation data"""
        try:
            if not validation_data:
                return float('-inf')
                
            model = self.models.get(timeframe)
            if not model:
                return float('-inf')
                
            # Split features and targets
            X = [sample['text'] for sample in validation_data]
            y = np.array([sample['changes'][timeframe] for sample in validation_data])
            
            # Check for valid target values
            if len(y) == 0 or np.all(np.isnan(y)):
                logger.warning(f"No valid target values for {timeframe}")
                return float('-inf')
            
            # Transform features
            X_tfidf = self.vectorizers[timeframe].transform(X)
            
            # Get predictions
            predictions = model.predict(X_tfidf)
            
            # Handle NaN values
            mask = ~np.isnan(predictions) & ~np.isnan(y)
            if not np.any(mask):
                logger.warning(f"No valid predictions for {timeframe}")
                return float('-inf')
                
            # Calculate score only on valid values
            score = -np.mean(np.abs(predictions[mask] - y[mask]))
            logger.info(f"{timeframe} model evaluation score: {score:.4f} (on {np.sum(mask)} valid samples)")
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating {timeframe} model: {str(e)}")
            return float('-inf')

    def train_with_impact_scores(self, training_data, timeframe):
        """Train model with impact scores and validate improvement"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data for {timeframe}")
                return False
                
            # Print training data statistics
            logger.info("\n" + "="*50)
            logger.info(f"TRAINING ANALYSIS FOR {timeframe}")
            logger.info("="*50)
            
            # Data statistics
            total_samples = len(training_data)
            unique_symbols = len(set(sample['symbol'] for sample in training_data))
            logger.info(f"\nTraining Data Stats:")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Unique symbols: {unique_symbols}")
            
            # Initialize models if they don't exist
            if timeframe not in self.vectorizers:
                logger.info(f"Initializing new TF-IDF vectorizer for {timeframe}")
                self.vectorizers[timeframe] = TfidfVectorizer(max_features=1000)
                
            if timeframe not in self.models:
                logger.info(f"Initializing new HybridMarketPredictor for {timeframe}")
                self.models[timeframe] = HybridMarketPredictor()
                
            # Split into train and validation
            train_size = int(len(training_data) * 0.8)
            train_data = training_data[:train_size]
            val_data = training_data[train_size:]
            
            # Get texts and semantic features for training
            X_train = [sample['text'] for sample in train_data]
            semantic_features = np.array([sample['embedding'] for sample in train_data])
            y_train = np.array([sample['changes'][timeframe] for sample in train_data])

            # Get current performance as baseline
            current_score = self.evaluate_model(timeframe, val_data)
            logger.info(f"\nBaseline Score: {current_score:.4f}")

            # Cluster Analysis
            logger.info("\nCluster Analysis:")
            try:
                # Define stop words to filter out common terms
                stop_words = {
                    # News sources
                    'Zacks', 'Reuters', 'Bloomberg', 'MarketWatch', 'Inc.', 'Corp.', 'LLC', 'Ltd',
                    # Common financial terms
                    'stock', 'stocks', 'share', 'shares', 'market', 'markets', 'trading', 'price', 'prices',
                    # Common words
                    'with', 'that', 'this', 'from', 'the', 'and', 'for', 'are', 'was', 'has', 'its', 'have',
                    'will', 'been', 'were', 'would', 'could', 'should', 'than', 'what', 'when', 'who',
                    'which', 'their', 'about', 'into', 'there', 'other', 'these', 'some', 'them',
                    # Common financial reporting terms
                    'quarter', 'quarterly', 'fiscal', 'year', 'annual', 'earnings', 'revenue', 'report',
                    'reported', 'financial', 'results'
                }
                
                # Perform clustering on semantic embeddings
                clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine').fit(semantic_features)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                logger.info(f"Initial clusters: {n_clusters}")
                logger.info(f"Noise points: {n_noise} ({(n_noise/len(labels))*100:.1f}%)")
                
                # Analyze each cluster
                cluster_stats = {}
                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue
                        
                    cluster_mask = labels == cluster_id
                    cluster_samples = [train_data[i] for i, is_member in enumerate(cluster_mask) if is_member]
                    cluster_changes = [sample['changes'][timeframe] for sample in cluster_samples if timeframe in sample['changes']]
                    cluster_changes = [c for c in cluster_changes if not np.isnan(c)]
                    
                    if cluster_changes:
                        # Basic cluster info
                        logger.info(f"\nCluster {cluster_id}:")
                        logger.info(f"  Size: {len(cluster_samples)} articles")
                        
                        # Stock analysis
                        cluster_symbols = set(s['symbol'] for s in cluster_samples)
                        logger.info(f"  Unique stocks: {len(cluster_symbols)}")
                        logger.info(f"  Symbols: {', '.join(cluster_symbols)}")
                        
                        # Price impact analysis
                        logger.info(f"  Price Impact Analysis:")
                        logger.info(f"    Mean change: {np.mean(cluster_changes):.2f}%")
                        logger.info(f"    Median change: {np.median(cluster_changes):.2f}%")
                        logger.info(f"    Std dev: {np.std(cluster_changes):.2f}%")
                        
                        # Sentiment analysis
                        cluster_sentiments = [s['sentiment']['score'] for s in cluster_samples]
                        pos_sent = len([s for s in cluster_sentiments if s > 0])
                        neg_sent = len([s for s in cluster_sentiments if s < 0])
                        logger.info(f"  Sentiment Analysis:")
                        logger.info(f"    Mean sentiment: {np.mean(cluster_sentiments):.3f}")
                        logger.info(f"    Positive articles: {pos_sent} ({pos_sent/len(cluster_sentiments)*100:.1f}%)")
                        logger.info(f"    Negative articles: {neg_sent} ({neg_sent/len(cluster_sentiments)*100:.1f}%)")
                        
                        # Content analysis
                        all_text = ' '.join(s['text'] for s in cluster_samples)
                        words = all_text.split()
                        word_freq = {}
                        for word in words:
                            if len(word) > 3 and word not in stop_words:  # Skip short words and stop words
                                word_freq[word] = word_freq.get(word, 0) + 1
                        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                        logger.info(f"  Content Analysis:")
                        logger.info(f"    Top terms: {', '.join(f'{word}({freq})' for word, freq in common_words)}")
                        
                        # Store stats for inter-cluster analysis
                        cluster_stats[cluster_id] = {
                            'size': len(cluster_samples),
                            'symbols': cluster_symbols,
                            'mean_change': np.mean(cluster_changes),
                            'mean_sentiment': np.mean(cluster_sentiments),
                            'common_words': [word for word, _ in common_words[:5]]
                        }
                
                # Inter-cluster analysis
                if len(cluster_stats) > 1:
                    logger.info("\nInter-cluster Relationships:")
                    for i, (c1_id, c1_stats) in enumerate(cluster_stats.items()):
                        for c2_id, c2_stats in list(cluster_stats.items())[i+1:]:
                            # Calculate similarities
                            common_symbols = c1_stats['symbols'] & c2_stats['symbols']
                            sentiment_diff = abs(c1_stats['mean_sentiment'] - c2_stats['mean_sentiment'])
                            price_impact_diff = abs(c1_stats['mean_change'] - c2_stats['mean_change'])
                            
                            # Only show meaningful relationships
                            if len(common_symbols) >= 2 or (sentiment_diff < 0.1 and price_impact_diff < 2.0):
                                logger.info(f"\n  Clusters {c1_id} and {c2_id} Relationship:")
                                if common_symbols:
                                    logger.info(f"    Common stocks: {', '.join(common_symbols)}")
                                logger.info(f"    Sentiment similarity: {1 - sentiment_diff:.2f}")
                                logger.info(f"    Price impact similarity: {100 - price_impact_diff:.1f}%")
                                
            except Exception as e:
                logger.error(f"Error in cluster analysis: {str(e)}")
            
            # Fit and transform the vectorizer
            logger.info(f"\nFitting TF-IDF vectorizer on {len(X_train)} samples")
            X_train_tfidf = self.vectorizers[timeframe].fit_transform(X_train)
            
            # Train the model with both TF-IDF and semantic features
            logger.info("Training HybridMarketPredictor with TF-IDF and semantic features")
            self.models[timeframe].fit(X_train_tfidf, y_train, semantic_features=semantic_features)
            
            # Evaluate new performance
            new_score = self.evaluate_model(timeframe, val_data)
            logger.info(f"\nNew Model Score: {new_score:.4f}")
            
            # Check if model improved
            if new_score > self.training_history[timeframe]['best_score'] + self.min_improvement:
                improvement = new_score - self.training_history[timeframe]['best_score']
                logger.info(f"\nModel improved by {improvement:.4f} points")
                self.training_history[timeframe]['best_score'] = new_score
                self.training_history[timeframe]['last_improvement'] = time.time()
                
                # Save models when there's improvement
                logger.info(f"Saving improved {timeframe} model...")
                try:
                    model_path = os.path.join(self.models_dir, f'market_model_{timeframe}.joblib')
                    vectorizer_path = os.path.join(self.models_dir, f'vectorizer_{timeframe}.joblib')
                    
                    joblib.dump(self.models[timeframe], model_path)
                    joblib.dump(self.vectorizers[timeframe], vectorizer_path)
                    
                    logger.info(f"Successfully saved {timeframe} model components")
                    
                    # Update backup with new data
                    self.training_data_backup[timeframe].extend(training_data)
                    if len(self.training_data_backup[timeframe]) > self.max_backup_size:
                        self.training_data_backup[timeframe] = self.training_data_backup[timeframe][-self.max_backup_size:]
                    
                except Exception as e:
                    logger.error(f"Error saving {timeframe} model: {str(e)}")
            else:
                logger.info("\nNo significant improvement, model not saved")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback
            return False

    def collect_and_train(self, symbols):
        """Collect news data and train models with checkpointing"""
        logger.info("="*80)
        logger.info("Starting training data collection...")
        logger.info(f"Total symbols to process: {len(symbols)}")
        logger.info("="*80)
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(checkpoint_dir, 'training_checkpoint.json')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    training_data, stock_stats, processed_symbols = self._restore_checkpoint_data(checkpoint)
                    logger.info(f"Loaded checkpoint with {len(processed_symbols)} processed symbols")
                    # Filter out already processed symbols
                    symbols = [s for s in symbols if s not in processed_symbols]
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                # Initialize fresh if checkpoint loading fails
                training_data, stock_stats, processed_symbols = self._initialize_fresh_data(symbols)
        else:
            # Initialize fresh training data and stats
            training_data, stock_stats, processed_symbols = self._initialize_fresh_data(symbols)
        
        # Split remaining symbols into chunks for API rate limiting
        symbol_chunks = [symbols[i:i+self.api_chunk_size] for i in range(0, len(symbols), self.api_chunk_size)]
        total_chunks = len(symbol_chunks)
        
        logger.info(f"Processing remaining {len(symbols)} symbols in {total_chunks} API chunks")
        logger.info(f"API chunk size: {self.api_chunk_size} symbols")
        logger.info(f"Delay between API chunks: {self.delay_between_chunks} seconds")
        
        # Process chunks sequentially with checkpointing
        for chunk_idx, chunk in enumerate(symbol_chunks, 1):
            try:
                logger.info(f"\nProcessing API chunk {chunk_idx}/{total_chunks}")
                logger.info(f"Symbols in this chunk: {', '.join(chunk)}")
                
                # Process each symbol in the chunk
                for symbol in chunk:
                    try:
                        logger.info(f"\nProcessing symbol: {symbol}")
                        samples = self.process_symbol(symbol)
                        stock_stats['processed'] += 1
                        
                        if samples:
                            stock_stats['with_news'] += 1
                            has_samples = False
                            for sample in samples:
                                for timeframe in training_data:
                                    if timeframe in sample['changes']:
                                        training_data[timeframe].append(sample)
                                        stock_stats['by_timeframe'][timeframe]['stocks'].add(symbol)
                                        stock_stats['by_timeframe'][timeframe]['samples'] += 1
                                        has_samples = True
                            
                            if has_samples:
                                processed_symbols.add(symbol)
                        else:
                            stock_stats['no_news'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {str(e)}")
                        stock_stats['failed'] += 1
                        continue
                
                # Train models after each chunk if we have enough data
                if any(len(data) >= 100 for data in training_data.values()):
                    logger.info("\nTraining models with current data...")
                    for timeframe, data in training_data.items():
                        if len(data) >= 100:  # Only train if we have enough samples
                            try:
                                if self.train_with_impact_scores(data, timeframe):
                                    # Only clear data if model improved and was saved
                                    training_data[timeframe] = []
                            except Exception as e:
                                logger.error(f"Error training {timeframe} model: {str(e)}")
                
                # Add delay between chunks unless it's the last chunk
                if chunk_idx < total_chunks:
                    logger.info(f"Waiting {self.delay_between_chunks} seconds before next chunk...")
                    time.sleep(self.delay_between_chunks)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                # Save checkpoint even if chunk fails
                checkpoint_data = self._prepare_checkpoint_data(
                    training_data=training_data,
                    stock_stats=stock_stats,
                    processed_symbols=processed_symbols
                )
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                continue
        
        # Final training with remaining data
        logger.info("\nFinal model training...")
        for timeframe, data in training_data.items():
            if data:  # Train with any remaining data
                try:
                    self.train_with_impact_scores(data, timeframe)
                except Exception as e:
                    logger.error(f"Error in final training for {timeframe}: {str(e)}")
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return stock_stats

    def _initialize_fresh_data(self, symbols):
        """Initialize fresh training data and statistics"""
        training_data = {'1wk': [], '1mo': []}
        stock_stats = {
            'total': len(symbols),
            'processed': 0,
            'failed': 0,
            'no_news': 0,
            'with_news': 0,
            'by_timeframe': {
                '1wk': {'stocks': set(), 'samples': 0},
                '1mo': {'stocks': set(), 'samples': 0}
            }
        }
        processed_symbols = set()
        return training_data, stock_stats, processed_symbols

    def get_symbols(self):
        """Get list of stock symbols to train on"""
        try:
            with open('./stock_tickers.txt', 'r') as file:
                symbols = file.read().strip().split('\n')
            logger.info(f"Processing {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.error(f"Error reading symbols file: {e}")
            return []
    
    def get_news_for_symbol(self, symbol):
        """Get news articles with optimized disk-based caching"""
        cache_file = os.path.join(self.cache_dir, f'{symbol}_news_cache.json')
        
        try:
            # Check disk cache
            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 3600:  # Cache valid for 1 hour
                    with open(cache_file, 'r') as f:
                        news_data = json.load(f)
                        return news_data
            
            # If not in cache, fetch new data
            search = yf.Search(
                query=symbol,
                news_count=20,
                include_nav_links=False,
                include_research=True
            )
            news = search.news
            
            # Write to cache file
            with open(cache_file, 'w') as f:
                json.dump(news, f)
            
            # Clear the search object
            del search
            
            return news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return []
        finally:
            gc.collect()

    def _process_article(self, article, symbol, stock):
        """Process a single news article and calculate its impact"""
        try:
            # Extract article data
            publish_time = article.get('providerPublishTime')
            if not publish_time:
                return None

            # Convert timestamp to datetime and ensure UTC
            publish_date = pd.Timestamp(publish_time, unit='s', tz='UTC')
            current_date = pd.Timestamp.now(tz='UTC')
            
            # Skip if article is too recent to have enough future data
            min_required_days = 35  # For monthly predictions
            if (current_date - publish_date).days < min_required_days:
                logger.info(f"Article for {symbol} is too recent, skipping")
                return None
            
            # Get article content
            title = article.get('title', '')
            summary = article.get('summary', '')
            content = f"{title} {summary}"
            
            if not content.strip():
                return None
                
            # Get price changes for different timeframes
            changes = {}
            try:
                # Get historical data with padding
                start_date = publish_date - pd.Timedelta(days=5)  # 5 days before for context
                end_date = publish_date + pd.Timedelta(days=40)   # 40 days after for monthly data
                
                # Get daily data
                history = stock.history(
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
                
                if len(history) < 2:
                    logger.warning(f"Insufficient price history for {symbol}")
                    return None
                
                # Find the closest price after publication date
                future_prices = history.loc[history.index >= publish_date]
                if len(future_prices) == 0:
                    logger.warning(f"No price data after publication date for {symbol}")
                    return None
                
                # Get the first available price after publication
                publish_price = future_prices.iloc[0]['Close']
                publish_actual_date = future_prices.index[0]
                
                # Calculate weekly change
                week_later = publish_actual_date + pd.Timedelta(days=7)
                week_data = history.loc[history.index >= week_later]
                if len(week_data) > 0:
                    week_price = week_data.iloc[0]['Close']
                    week_actual_date = week_data.index[0]
                    # Only include if within reasonable range (5-9 days)
                    days_diff = (week_actual_date - publish_actual_date).days
                    if 5 <= days_diff <= 9:
                        changes['1wk'] = ((week_price - publish_price) / publish_price) * 100
                
                # Calculate monthly change
                month_later = publish_actual_date + pd.Timedelta(days=30)
                month_data = history.loc[history.index >= month_later]
                if len(month_data) > 0:
                    month_price = month_data.iloc[0]['Close']
                    month_actual_date = month_data.index[0]
                    # Only include if within reasonable range (28-32 days)
                    days_diff = (month_actual_date - publish_actual_date).days
                    if 28 <= days_diff <= 32:
                        changes['1mo'] = ((month_price - publish_price) / publish_price) * 100
                
                if not changes:
                    logger.info(f"No valid price changes within acceptable ranges for {symbol}")
                    return None
                
            except Exception as e:
                logger.error(f"Error calculating price changes for {symbol}: {str(e)}")
                return None
            
            # Get sentiment analysis
            sentiment = self.finbert_analyzer.analyze_sentiment(content)
            if not sentiment:
                return None
            
            # Get semantic embedding
            embedding = self.semantic_analyzer.get_embedding(content)
            if embedding is None:
                return None
            
            # Create training sample
            sample = {
                'symbol': symbol,
                'text': content,
                'publish_date': publish_date.isoformat(),
                'sentiment': sentiment,
                'embedding': embedding.tolist(),
                'changes': changes
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing article for {symbol}: {str(e)}")
            return None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def analyze_stock(self, stock, publish_time):
        """Analyze stock price changes after news publication"""
        changes = {}
        
        try:
            publish_date = pd.Timestamp(publish_time, unit='s', tz='UTC')
            current_date = pd.Timestamp.now(tz='UTC')
            
            logger.info(f"\nAnalyzing price changes:")
            logger.info(f"  Article Date: {publish_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"  Current Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            for timeframe, params in self.TIMEFRAMES.items():
                try:
                    future_date = publish_date + pd.Timedelta(days=params['days'])
                    logger.info(f"\n{timeframe} Analysis:")
                    logger.info(f"  Start Date: {publish_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    logger.info(f"  Target End Date: {future_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    
                    # Check if we're trying to use future data
                    days_into_future = (future_date - current_date).total_seconds() / (24 * 3600)
                    
                    if days_into_future > 0:
                        logger.warning(f"  SKIPPING - Would need {days_into_future:.1f} days of future data")
                        continue
                    
                    # For weekly and monthly timeframes, use daily data
                    prices = stock.history(
                        start=publish_date,
                        end=future_date,
                        interval=params['interval']
                    )
                    
                    if len(prices) >= params['min_points']:
                        start_price = prices['Close'].iloc[0]
                        end_price = prices['Close'].iloc[-1]
                        percent_change = ((end_price - start_price) / start_price) * 100
                        
                        actual_start_date = prices.index[0]
                        actual_end_date = prices.index[-1]
                        
                        logger.info(f"  Actual Data Range:")
                        logger.info(f"    Start: {actual_start_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        logger.info(f"    End: {actual_end_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        logger.info(f"    Start Price: ${start_price:.2f}")
                        logger.info(f"    End Price: ${end_price:.2f}")
                        logger.info(f"    Percent Change: {percent_change:.2f}%")
                        logger.info(f"    Data Points: {len(prices)}")
                        
                        # Double check that we're not using future data
                        if actual_end_date > current_date:
                            logger.warning(f"  SKIPPING - Last price date {actual_end_date} is in the future")
                            continue
                            
                        changes[timeframe] = percent_change
                    else:
                        logger.info(f"  Insufficient data points: {len(prices)} < {params['min_points']}")
                
                except Exception as e:
                    logger.warning(f"  Error calculating {timeframe} change: {str(e)}")
                    continue
            
            if not changes:
                logger.info("No valid price changes found for any timeframe")
            else:
                logger.info("\nValid changes recorded:")
                for tf, change in changes.items():
                    logger.info(f"  {tf}: {change:.2f}%")
            
            return changes
            
        except Exception as e:
            logger.error(f"Error in analyze_stock: {str(e)}")
            return {}

    def get_full_article_text(self, url):
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

    def _prepare_checkpoint_data(self, training_data, stock_stats, processed_symbols):
        """Prepare data for checkpoint saving"""
        # Convert sets to lists for JSON serialization
        serializable_stats = stock_stats.copy()
        serializable_stats['by_timeframe'] = {
            timeframe: {
                'stocks': list(data['stocks']),
                'samples': data['samples']
            }
            for timeframe, data in stock_stats['by_timeframe'].items()
        }
        
        return {
            'training_data': training_data,
            'stock_stats': serializable_stats,
            'processed_symbols': list(processed_symbols)
        }
    
    def _restore_checkpoint_data(self, checkpoint):
        """Restore data from checkpoint"""
        try:
            training_data = checkpoint['training_data']
            
            # Restore stock stats with sets
            stock_stats = checkpoint['stock_stats']
            stock_stats['by_timeframe'] = {
                timeframe: {
                    'stocks': set(data['stocks']),
                    'samples': data['samples']
                }
                for timeframe, data in stock_stats['by_timeframe'].items()
            }
            
            # Restore processed symbols set
            processed_symbols = set(checkpoint['processed_symbols'])
            
            return training_data, stock_stats, processed_symbols
            
        except Exception as e:
            logger.error(f"Error restoring checkpoint: {str(e)}")
            # Return fresh data if restoration fails
            return self._initialize_fresh_data([])

def main():
    """Main entry point for training"""
    logger.info("Starting Market ML Training...")
    trainer = MarketMLTrainer()
    trainer.train()

if __name__ == "__main__":
    main()