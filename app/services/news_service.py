import aiohttp
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from ..models.news import NewsArticle
import logging
from bs4 import BeautifulSoup, SoupStrainer
import time
import yfinance as yf
import gc
from functools import lru_cache

logger = logging.getLogger(__name__)

class NewsService:
    def __init__(self):
        self.base_url = "https://finance.yahoo.com"
        self.last_request_time = {}  # Track last request time per symbol
        self.min_request_interval = 2  # Minimum seconds between requests
        self.subreddits = ['wallstreetbets', 'stocks', 'investing']
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self.session = None
        self.news_table_strainer = SoupStrainer('table', {'class': 'news-table'})

    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()

    @lru_cache(maxsize=100)
    def get_headers(self):
        """Cache headers to save memory"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def fetch_stock_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news from multiple sources"""
        try:
            # Fetch news from sources one at a time to reduce memory usage
            all_news = []
            
            # Yahoo Finance news (primary source)
            yahoo_news = await self._fetch_yahoo_news(symbol)
            all_news.extend(yahoo_news)
            
            # Only fetch from additional sources if we don't have enough news
            if len(all_news) < 5:
                finviz_news = await self._fetch_finviz_news(symbol)
                all_news.extend(finviz_news)
                
                if len(all_news) < 5:
                    reddit_news = await self._fetch_reddit_posts(symbol)
                    all_news.extend(reddit_news)
            
            # Sort by published date, newest first
            all_news.sort(key=lambda x: x.published_at, reverse=True)
            
            # Limit the number of articles to reduce memory usage
            return all_news[:10]
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
        finally:
            # Force garbage collection
            gc.collect()

    async def _fetch_finviz_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news from Finviz"""
        try:
            async with self.semaphore:  # Limit concurrent requests
                url = f"https://finviz.com/quote.ashx?t={symbol.lower()}"
                session = await self.get_session()
                
                async with session.get(url, headers=self.get_headers()) as response:
                    if response.status != 200:
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser', parse_only=self.news_table_strainer)
                    news_table = soup.find('table', {'class': 'news-table'})
                    if not news_table:
                        return []
                    
                    articles = []
                    current_date = None
                    today = datetime.now(timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    week_ago = today - timedelta(days=7)
                    
                    for row in news_table.find_all('tr', limit=20):  # Limit number of articles
                        cells = row.find_all('td')
                        if len(cells) == 2:
                            date_cell = cells[0].text.strip()
                            
                            try:
                                if 'Today' in date_cell:
                                    time_str = date_cell.replace('Today ', '')
                                    current_date = today
                                    
                                    is_pm = time_str.lower().endswith('pm')
                                    time_str = time_str.lower().replace('am', '').replace('pm', '')
                                    hour, minute = map(int, time_str.split(':'))
                                    
                                    if is_pm and hour != 12:
                                        hour += 12
                                    elif not is_pm and hour == 12:
                                        hour = 0
                                        
                                    current_date = current_date.replace(hour=hour, minute=minute)
                                    
                                elif len(date_cell) > 8:
                                    current_date = datetime.strptime(date_cell, '%b-%d-%y').replace(
                                        hour=16, minute=0, second=0, microsecond=0
                                    )
                                else:
                                    if current_date is None:
                                        continue
                                    
                                    time_str = date_cell.lower()
                                    is_pm = 'pm' in time_str
                                    time_str = time_str.replace('am', '').replace('pm', '')
                                    hour, minute = map(int, time_str.split(':'))
                                    
                                    if is_pm and hour != 12:
                                        hour += 12
                                    elif not is_pm and hour == 12:
                                        hour = 0
                                        
                                    current_date = current_date.replace(hour=hour, minute=minute)
                                
                                if current_date and current_date.replace(tzinfo=timezone.utc) > week_ago:
                                    title = cells[1].text.strip()
                                    link = cells[1].a['href']
                                    
                                    articles.append(NewsArticle(
                                        title=title,
                                        url=link,
                                        source="Finviz",
                                        content=title,
                                        published_at=current_date.replace(tzinfo=timezone.utc),
                                        related_symbols=[symbol],
                                        sentiment_score=None
                                    ))
                                    
                            except Exception as e:
                                logger.debug(f"Skipping Finviz news item for {symbol} due to date parsing: {str(e)}")
                                continue
                    
                    return articles[:5]  # Limit to 5 most recent articles
                    
        except Exception as e:
            logger.error(f"Error fetching Finviz news for {symbol}: {str(e)}")
            return []

    async def _fetch_reddit_posts(self, symbol: str) -> List[NewsArticle]:
        """Fetch relevant posts from Reddit using web scraping"""
        try:
            async with self.semaphore:  # Limit concurrent requests
                articles = []
                week_ago = datetime.now(timezone.utc) - timedelta(days=7)
                session = await self.get_session()
                
                for subreddit in self.subreddits:
                    url = f"https://www.reddit.com/r/{subreddit}/search/.json?q={symbol}&restrict_sr=1&sort=new&t=week&limit=5"
                    
                    try:
                        async with session.get(url, headers=self.get_headers()) as response:
                            if response.status != 200:
                                continue
                                
                            data = await response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            for post in posts[:3]:  # Limit posts per subreddit
                                post_data = post.get('data', {})
                                title = post_data.get('title', '')
                                
                                if symbol.upper() in title.upper():
                                    published_at = datetime.fromtimestamp(
                                        post_data.get('created_utc', 0), 
                                        tz=timezone.utc
                                    )
                                    
                                    if published_at > week_ago:
                                        articles.append(NewsArticle(
                                            title=title,
                                            url=f"https://reddit.com{post_data.get('permalink')}",
                                            source=f"Reddit/r/{subreddit}",
                                            content=post_data.get('selftext', '')[:500],  # Limit content length
                                            published_at=published_at,
                                            related_symbols=[symbol],
                                            sentiment_score=None
                                        ))
                        
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error fetching from r/{subreddit}: {str(e)}")
                        continue
                
                return articles[:5]  # Limit to 5 most relevant posts
                
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {symbol}: {str(e)}")
            return []

    async def _fetch_yahoo_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        try:
            async with self.semaphore:  # Limit concurrent requests
                now = time.time()
                if symbol in self.last_request_time:
                    time_since_last = now - self.last_request_time[symbol]
                    if time_since_last < self.min_request_interval:
                        await asyncio.sleep(self.min_request_interval - time_since_last)
                
                self.last_request_time[symbol] = now

                for attempt in range(3):
                    try:
                        # Create a new Ticker instance for each attempt
                        stock = yf.Ticker(symbol)
                        
                        # Try to get news with error handling
                        try:
                            news_data = stock.news
                            if news_data and isinstance(news_data, list):
                                return self._process_news_data(news_data[:10], symbol)  # Limit to 10 articles
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to get news for {symbol} on attempt {attempt + 1}: {str(e)}")
                        
                        # Try alternative method: get info first
                        try:
                            info = stock.info
                            if info:  # If we can get info, try news again after a short delay
                                await asyncio.sleep(1)
                                news_data = stock.news
                                if news_data and isinstance(news_data, list):
                                    return self._process_news_data(news_data[:10], symbol)
                        except Exception as e:
                            logger.debug(f"Failed to get stock info for {symbol} on attempt {attempt + 1}: {str(e)}")
                        
                        if attempt < 2:
                            await asyncio.sleep(2 ** (attempt + 1))  # Exponential backoff
                            
                    except Exception as e:
                        logger.debug(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** (attempt + 1))
                        else:
                            logger.error(f"All attempts failed for {symbol}")

                return []

        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {symbol}: {str(e)}")
            return []

    def _process_news_data(self, news_data: List[dict], symbol: str) -> List[NewsArticle]:
        """Process news data into NewsArticle objects"""
        articles = []
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        try:
            for item in news_data[:5]:  # Limit to 5 most recent articles
                try:
                    # Validate required fields
                    if not all(k in item for k in ['providerPublishTime', 'title', 'link']):
                        continue
                        
                    published_at = datetime.fromtimestamp(item['providerPublishTime'], tz=timezone.utc)
                    
                    if published_at > week_ago:
                        articles.append(NewsArticle(
                            title=item['title'],
                            url=item['link'],
                            source=item.get('publisher', 'Yahoo Finance'),
                            content=item.get('summary', '')[:500],  # Limit content length
                            published_at=published_at,
                            related_symbols=[symbol],
                            sentiment_score=None
                        ))
                except Exception as e:
                    logger.debug(f"Error processing news item for {symbol}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing news data for {symbol}: {str(e)}")
            
        return articles 