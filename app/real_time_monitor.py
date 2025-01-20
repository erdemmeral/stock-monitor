from app.models.stock_analyzer import StockAnalyzer


class StockMonitor:
    def __init__(self):
        self.stock_analyzer = StockAnalyzer()  # Initialize ML-enabled analyzer    
    def process_stock(self, symbol):
        news = self.get_news(symbol)
        # This will now use ML-based sentiment scoring
        sentiment = self.stock_analyzer.process_news(news)
import asyncio
import time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, List
import random

# Filter out yfinance TimedeltaIndex deprecation warning
warnings.filterwarnings('ignore', message='.*The \'unit\' keyword in TimedeltaIndex.*')

from app.services.news_service import NewsService
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.stock_analyzer import StockAnalyzer
from app.utils.sp500 import get_all_symbols
from app.models.stock import StockAnalysis
from app.models.news import NewsArticle
from app.utils.telegram_notifier import TelegramNotifier
from aiohttp import web
import os
import logging
from app.models.position import Position
from app.models.portfolio import Portfolio
import aiohttp
import yfinance as yf

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

class StockMonitor:
    def __init__(self):
        self.news_service = NewsService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stock_analyzer = StockAnalyzer()
        self.notifier = TelegramNotifier()
        self.notifier.set_monitor(self)  # Set monitor reference for command handling
        self.last_analysis: Dict[str, StockAnalysis] = {}
        self.processed_news_ids: Dict[str, Set[str]] = {}
        self.update_interval = 300  # 5 minutes
        self.last_heartbeat = datetime.now(timezone.utc)
        self.is_running = True
        self.active_buy_signals: Set[str] = set()
        self.portfolio = Portfolio()
        self.last_market_status = None  # Track last market status
        self.holiday_weekend_notified = False  # Track if holiday/weekend notification was sent
        self.last_market_state = None  # Track previous market state

    def get_et_time(self):
        """Get current Eastern Time (handling both EST and EDT)"""
        utc_time = datetime.now(timezone.utc)
        # Use US/Eastern to automatically handle DST
        eastern = timezone(timedelta(hours=-5))  # EST base offset
        
        # Check if we're in DST
        # DST starts second Sunday in March and ends first Sunday in November
        year = utc_time.year
        dst_start = datetime(year, 3, 8, 2, 0, tzinfo=eastern)  # 2 AM on second Sunday in March
        while dst_start.weekday() != 6:  # Adjust to Sunday
            dst_start += timedelta(days=1)
            
        dst_end = datetime(year, 11, 1, 2, 0, tzinfo=eastern)  # 2 AM on first Sunday in November
        while dst_end.weekday() != 6:  # Adjust to Sunday
            dst_end += timedelta(days=1)
            
        if dst_start <= utc_time.astimezone(eastern) < dst_end:
            eastern = timezone(timedelta(hours=-4))  # EDT
            
        return utc_time.astimezone(eastern)

    async def monitor_stock(self, symbol: str):
        """Monitor a single stock."""
        failures = 0
        max_failures = 3
        
        try:
            current_time = self.get_et_time()
            
            logger.info(f"Fetching news for {symbol}")
            try:
                news_articles = await self.news_service.fetch_stock_news(symbol)
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    logger.warning(f"Rate limit hit, taking a 5-minute break...")
                    await asyncio.sleep(300)  # 5-minute break
                    return
                raise e
            
            if symbol not in self.processed_news_ids:
                self.processed_news_ids[symbol] = set()
            
            new_articles = [
                article for article in news_articles 
                if article.url not in self.processed_news_ids[symbol]
            ]

            if new_articles:
                logger.info(f"Found {len(new_articles)} new articles for {symbol}")
                
                for article in new_articles:
                    article.sentiment_score = self.sentiment_analyzer.analyze_sentiment(article)
                    self.processed_news_ids[symbol].add(article.url)
                    logger.info(f"New Article for {symbol}: [{article.source}] {article.title} (Sentiment: {article.sentiment_score:.2f})")
                    
                    # Check for compliance regaining news
                    title_lower = article.title.lower()
                    content_lower = article.content.lower() if article.content else ""
                    compliance_keywords = [
                        "regains compliance", "regained compliance",
                        "regains nasdaq compliance", "regained nasdaq compliance",
                        "regains nyse compliance", "regained nyse compliance",
                        "compliance requirements met", "meets compliance requirements",
                        "compliance notice", "notice of compliance"
                    ]
                    
                    if any(keyword in title_lower or keyword in content_lower for keyword in compliance_keywords):
                        await self.notifier.send_message(
                            f"🎯 <b>Compliance Alert for {symbol}</b>\n\n"
                            f"Source: {article.source}\n"
                            f"Title: {article.title}\n"
                            f"Time: {article.published_at.strftime('%I:%M %p ET') if article.published_at else 'N/A'}\n"
                            f"URL: {article.url}\n\n"
                            f"This could be a significant positive development for the stock."
                        )
                        logger.info(f"🎯 Compliance regaining alert sent for {symbol}")
                
                analysis = self.stock_analyzer.analyze_stock(symbol, news_articles)
                
                if analysis:
                    prev_analysis = self.last_analysis.get(symbol)
                    self.last_analysis[symbol] = analysis
                    
                    # Update position if exists
                    if symbol in self.portfolio.positions:
                        position = self.portfolio.positions[symbol]
                        position.current_price = analysis.current_price
                        position.current_sentiment = analysis.sentiment_score
                        position.current_confidence = analysis.confidence_score
                        position.last_updated = analysis.analysis_timestamp
                    
                    # Check for buy signal
                    if analysis.should_buy() and symbol not in self.active_buy_signals:
                        relevant_news = [
                            article for article in new_articles
                            if article.sentiment_score > 0.3
                        ]
                        await self._notify_buy_signal(analysis, relevant_news)
                        self.active_buy_signals.add(symbol)
                        position = Position(
                            symbol=symbol,
                            entry_price=analysis.current_price,
                            entry_time=analysis.analysis_timestamp,
                            entry_sentiment=analysis.sentiment_score,
                            entry_confidence=analysis.confidence_score,
                            current_price=analysis.current_price,
                            current_sentiment=analysis.sentiment_score,
                            current_confidence=analysis.confidence_score,
                            last_updated=analysis.analysis_timestamp
                        )
                        self.portfolio.add_position(position)
                        logger.info(f"Added {symbol} to portfolio at ${analysis.current_price:.2f}")
                    
                    # Check for sell signal
                    elif symbol in self.active_buy_signals:
                        if analysis.should_sell(prev_analysis):
                            negative_news = [
                                article for article in new_articles
                                if article.sentiment_score < -0.2
                            ]
                            position = self.portfolio.positions.get(symbol)
                            self.portfolio.close_position(
                                symbol,
                                analysis.current_price,
                                analysis.analysis_timestamp
                            )
                            await self._notify_sell_signal(analysis, prev_analysis, position, negative_news)
                            self.active_buy_signals.remove(symbol)
                        elif analysis.confidence_score > prev_analysis.confidence_score:
                            await self._notify_confidence_increase(analysis, prev_analysis)
            else:
                logger.debug(f"No new articles found for {symbol}")
            
            # Reset failures on success
            failures = 0
            
            # Add small delay to avoid overwhelming API
            await asyncio.sleep(0.5)
            
        except Exception as e:
            failures += 1
            logger.error(f"Error monitoring {symbol} (attempt {failures}/{max_failures}): {str(e)}")
            if failures >= max_failures:
                logger.warning(f"Temporarily suspending monitoring of {symbol}")
                await asyncio.sleep(self.update_interval * 2)
                failures = 0

    async def _notify_buy_signal(self, analysis: StockAnalysis, relevant_news: List[NewsArticle]):
        """Notify user of a new buy signal with relevant news links."""
        try:
            await self.notifier.send_buy_signal(analysis, relevant_news)
            logger.info(f"Buy signal sent for {analysis.symbol}")
        except Exception as e:
            logger.error(f"Failed to send buy signal for {analysis.symbol}: {str(e)}")

    async def _notify_confidence_increase(self, new_analysis: StockAnalysis, prev_analysis: StockAnalysis):
        """Notify user of increased buy confidence."""
        try:
            await self.notifier.send_confidence_update(new_analysis, prev_analysis)
            logger.info(f"Confidence update sent for {new_analysis.symbol}")
        except Exception as e:
            logger.error(f"Error sending confidence update: {str(e)}")

    async def _notify_sell_signal(self, analysis: StockAnalysis, prev_analysis: StockAnalysis, 
                                position: Optional[Position] = None, negative_news: List[NewsArticle] = None):
        """Notify user of a sell signal with position details and negative news if available."""
        try:
            # Get detailed sell reasons
            should_sell, sell_reasons = analysis.should_sell(prev_analysis)
            if not should_sell:
                return
                
            # Format sell reasons
            reasons_text = "\n".join(f"• {reason}" for reason in sell_reasons)
            
            # Build message
            message = [
                f"🔴 <b>SELL SIGNAL: {analysis.symbol}</b>\n",
                f"Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n",
                f"Current Price: ${analysis.current_price:.2f}\n",
                f"Previous Price: ${prev_analysis.current_price:.2f}\n",
                f"Current Sentiment: {analysis.sentiment_score:.2f}\n",
                f"Previous Sentiment: {prev_analysis.sentiment_score:.2f}\n",
                f"Confidence: {analysis.confidence_score:.2f}\n",
                "\n<b>Sell Reasons:</b>\n" + reasons_text
            ]
            
            # Add position details if available
            if position:
                profit_loss = ((analysis.current_price - position.entry_price) / position.entry_price) * 100
                message.extend([
                    "\n<b>Position Summary:</b>",
                    f"Holding Duration: {position.holding_duration:.1f} days",
                    f"Entry Price: ${position.entry_price:.2f}",
                    f"Total P/L: {profit_loss:.2f}%",
                    f"Sentiment Change: {analysis.sentiment_score - position.entry_sentiment:.2f}"
                ])
            
            # Add relevant negative news if available
            if negative_news:
                message.append("\n<b>Recent Negative News:</b>")
                for article in negative_news[:3]:  # Show up to 3 negative articles
                    time_ago = self._format_time_ago(article.published_at)
                    message.append(
                        f"📰 {article.title}\n"
                        f"  ({article.source}, {time_ago}, Sentiment: {article.sentiment_score:.2f})"
                    )
            
            await self.notifier.send_message("\n".join(message))
            logger.info(f"Sell signal sent for {analysis.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to send sell signal for {analysis.symbol}: {str(e)}")

    async def _notify_portfolio_update(self):
        """Send portfolio summary update"""
        try:
            summary = self.portfolio.get_portfolio_summary()
            await self.notifier.send_message(
                f"📊 <b>Portfolio Update</b>\n{summary}"
            )
        except Exception as e:
            logger.error(f"Failed to send portfolio update: {str(e)}")

    async def handle_telegram_update(self, request):
        """Handle incoming Telegram updates"""
        try:
            data = await request.json()
            logger.info(f"Received Telegram update: {data}")
            
            if 'message' in data:
                message = data['message']
                
                # Get chat ID and message text
                chat_id = str(message.get('chat', {}).get('id'))
                message_text = message.get('text', '')
                
                # Check if it's a command
                if message_text.startswith('/'):
                    # Extract command and any mention (like @bot_name)
                    command_parts = message_text.split('@', 1)
                    command = command_parts[0]  # Get the actual command
                    
                    # If there's a mention, verify it's for our bot
                    if len(command_parts) > 1:
                        bot_mention = command_parts[1].lower()
                        # Get our bot's username from the token (last part after :)
                        our_bot_name = self.notifier.bot_token.split(':')[-1].lower()
                        if bot_mention and bot_mention != our_bot_name:
                            # Command is for a different bot
                            return web.Response(text='OK')
                    
                    # Verify the message is from the configured chat
                    if chat_id != str(self.notifier.chat_id):
                        logger.warning(f"Received message from unauthorized chat: {chat_id}")
                        return web.Response(text='Unauthorized chat', status=403)
                    
                    logger.info(f"Processing command: {command}")
                    response = await self.notifier.handle_command(command)
                    if response:
                        logger.info(f"Sending response for command {command}")
                        await self.notifier.send_message(response)
                    else:
                        logger.warning(f"No response generated for command: {command}")
                        
            return web.Response(text='OK')
            
        except Exception as e:
            logger.error(f"Error handling Telegram update: {str(e)}")
            return web.Response(text='Error', status=500)

    async def handle_health_check(self, request):
        """Handle health check requests"""
        return web.Response(text='Stock Monitor is running', status=200)

    async def start_web_server(self):
        """Start web server for Telegram webhook"""
        port = int(os.getenv('PORT', 8080))
        max_retries = 3
        retry_count = 0
        
        # Set up the webhook URL for Telegram
        webhook_url = f"https://api.telegram.org/bot{self.notifier.bot_token}/setWebhook"
        render_url = "https://stock-monitor-latest.onrender.com"
        
        while retry_count < max_retries:
            try:
                app = web.Application()
                app.router.add_post('/webhook', self.handle_telegram_update)
                app.router.add_get('/', self.handle_health_check)  # Add health check route
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, '0.0.0.0', port)
                await site.start()
                logger.info(f"Web server started on port {port}")
                
                # Set the webhook
                async with aiohttp.ClientSession() as session:
                    webhook_data = {
                        "url": f"{render_url}/webhook",
                        "allowed_updates": ["message"]
                    }
                    async with session.post(webhook_url, json=webhook_data) as response:
                        result = await response.json()
                        if response.status == 200 and result.get('ok'):
                            logger.info(f"Telegram webhook set successfully to {webhook_data['url']}")
                        else:
                            logger.error(f"Failed to set Telegram webhook: {result}")
                            # Try to get webhook info for debugging
                            webhook_info_url = f"https://api.telegram.org/bot{self.notifier.bot_token}/getWebhookInfo"
                            async with session.get(webhook_info_url) as info_response:
                                info = await info_response.json()
                                logger.info(f"Current webhook info: {info}")
                
                return
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    retry_count += 1
                    if retry_count < max_retries:
                        port += 1
                        logger.warning(f"Port {port-1} in use, trying port {port}")
                    else:
                        raise RuntimeError(f"Could not find available port after {max_retries} attempts")
                else:
                    raise

    def is_market_open(self) -> bool:
        """Check if the market is currently open, including pre-market and after-hours"""
        now = datetime.now(timezone.utc)
        
        # Convert UTC to Eastern Time (US Market)
        et_time = now.astimezone(timezone(timedelta(hours=-4)))  # EST/EDT
        
        # Reset holiday/weekend notification flag if it's a trading day
        if et_time.weekday() < 5 and et_time.strftime('%Y-%m-%d') not in [
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
            "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
            "2024-11-28", "2024-12-25"
        ]:
            self.holiday_weekend_notified = False
        
        # Check if it's a weekday
        if et_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            current_status = f"Weekend ({['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][et_time.weekday()]})"
            if current_status != self.last_market_status:
                logger.info(f"🌙 Market closed - {current_status}")
                self.last_market_status = current_status
                
                # Send weekend notification if not sent yet
                if not self.holiday_weekend_notified:
                    asyncio.create_task(self._notify_market_closed(
                        f"Market is closed for the weekend. Trading will resume on "
                        f"{(et_time + timedelta(days=7-et_time.weekday())).strftime('%A, %B %d')}"
                    ))
                    self.holiday_weekend_notified = True
            return False
            
        # Major US Market Holidays
        holidays_2024 = {
            "New Year's Day": "2024-01-01",
            "Martin Luther King Jr. Day": "2024-01-15",
            "Presidents Day": "2024-02-19",
            "Good Friday": "2024-03-29",
            "Memorial Day": "2024-05-27",
            "Juneteenth": "2024-06-19",
            "Independence Day": "2024-07-04",
            "Labor Day": "2024-09-02",
            "Thanksgiving": "2024-11-28",
            "Christmas": "2024-12-25"
        }
        
        # Check if today is a holiday
        today_str = et_time.strftime('%Y-%m-%d')
        if today_str in holidays_2024.values():
            holiday_name = [k for k, v in holidays_2024.items() if v == today_str][0]
            current_status = f"Holiday ({holiday_name})"
            if current_status != self.last_market_status:
                logger.info(f"🌙 Market closed - {current_status}")
                self.last_market_status = current_status
                
                # Send holiday notification if not sent yet
                if not self.holiday_weekend_notified:
                    # Find next trading day
                    next_day = et_time + timedelta(days=1)
                    while (next_day.weekday() >= 5 or 
                           next_day.strftime('%Y-%m-%d') in holidays_2024.values()):
                        next_day += timedelta(days=1)
                    
                    asyncio.create_task(self._notify_market_closed(
                        f"Market is closed for {holiday_name}. Trading will resume on "
                        f"{next_day.strftime('%A, %B %d')}"
                    ))
                    self.holiday_weekend_notified = True
            return False
        
        # Define trading hours
        premarket_start = et_time.replace(hour=4, minute=0, second=0, microsecond=0)
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        afterhours_end = et_time.replace(hour=20, minute=0, second=0, microsecond=0)
        
        # Determine market session
        is_open = True
        
        if premarket_start <= et_time < market_open:
            current_status = "Pre-market (4:00 AM - 9:30 AM ET)"
        elif market_open <= et_time <= market_close:
            current_status = "Regular Trading (9:30 AM - 4:00 PM ET)"
        elif market_close < et_time <= afterhours_end:
            current_status = "After-hours (4:00 PM - 8:00 PM ET)"
        else:
            current_status = "Outside Trading Hours (8:00 PM - 4:00 AM ET)"
            is_open = False
        
        # Log status change
        if current_status != self.last_market_status:
            if is_open:
                logger.info(f"🔔 Market session: {current_status}")
            else:
                logger.info(f"🌙 Market closed - {current_status}")
            self.last_market_status = current_status
        
        return is_open

    async def sleep_until_market_open(self):
        """Sleep until the next available trading session"""
        now = datetime.now(timezone.utc)
        et_time = now.astimezone(timezone(timedelta(hours=-4)))
        
        # Calculate next trading session start time
        premarket_start = et_time.replace(hour=4, minute=0, second=0, microsecond=0)
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        afterhours_end = et_time.replace(hour=20, minute=0, second=0, microsecond=0)
        
        # Determine next session start
        if et_time.hour < 4:  # Before pre-market
            next_session = premarket_start
            session_type = "pre-market"
        elif et_time.hour < 9 or (et_time.hour == 9 and et_time.minute < 30):  # Before regular market
            next_session = market_open
            session_type = "regular market"
        elif et_time.hour >= 20:  # After all sessions
            # Set to next day's pre-market
            next_session = premarket_start + timedelta(days=1)
            session_type = "next day pre-market"
        else:
            next_session = premarket_start + timedelta(days=1)
            session_type = "next day pre-market"
        
        # Keep adjusting until we find a valid market day
        original_next_session = next_session
        while True:
            # Skip weekends
            while next_session.weekday() >= 5:
                next_session += timedelta(days=1)
            
            # Skip holidays
            if next_session.strftime('%Y-%m-%d') in [
                "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
                "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
                "2024-11-28", "2024-12-25"
            ]:
                next_session += timedelta(days=1)
                continue
            
            break
        
        sleep_seconds = (next_session - et_time).total_seconds()
        if sleep_seconds > 0:
            # Calculate sleep duration in a readable format
            sleep_hours = sleep_seconds // 3600
            sleep_minutes = (sleep_seconds % 3600) // 60
            
            logger.info(
                f"💤 Entering sleep mode - Market is closed\n"
                f"Current time: {et_time.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
                f"Next session: {next_session.strftime('%Y-%m-%d %H:%M:%S ET')} ({session_type})\n"
                f"Sleep duration: {sleep_hours:.0f}h {sleep_minutes:.0f}m"
            )
            
            await asyncio.sleep(sleep_seconds)
            
            logger.info(
                f"⏰ Waking up - {session_type.title()} session starting\n"
                f"Current time: {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d %H:%M:%S ET')}"
            )

    async def update_positions(self):
        """Update current prices and holding times for all positions"""
        last_update_time = None  # Track when we last sent a summary (HH:MM)
        api_calls = {}  # Track API calls per symbol
        last_session = None  # Track last market session
        
        while self.is_running:
            try:
                current_time = self.get_et_time()
                
                # Determine market session
                if not self.is_market_open():
                    current_session = "Market Closed"
                else:
                    # Only check specific sessions when market is open
                    if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                        current_session = "Pre-market (4:00 AM - 9:30 AM ET)"
                    elif current_time.hour < 16 or (current_time.hour == 16 and current_time.minute == 0):
                        current_session = "Regular Trading Hours (9:30 AM - 4:00 PM ET)"
                    elif current_time.hour < 20:
                        current_session = "After-hours (4:00 PM - 8:00 PM ET)"
                    else:
                        current_session = "Market Closed"
                
                # Notify on session change (only once)
                if current_session != last_session:
                    logger.info(f"Market session changed to: {current_session}")
                    await self.notifier.send_message(
                        f"🔔 Market Session Change: {current_session}\n"
                        f"Current Time: {current_time.strftime('%I:%M %p ET')}"
                    )
                    last_session = current_session
                
                # Check if any trading session is open
                if not self.is_market_open():
                    await self.sleep_until_market_open()
                    continue
                
                if self.portfolio.positions:
                    # Send summary only at market open (9:30 AM) and market close (4:00 PM)
                    is_market_open_time = current_time.hour == 9 and current_time.minute == 30
                    is_market_close_time = current_time.hour == 16 and current_time.minute == 0
                    
                    # Only update if we haven't sent an update at this exact minute
                    current_key = f"{current_time.hour}:{current_time.minute}"
                    should_send_update = (
                        (is_market_open_time or is_market_close_time) and 
                        current_key != last_update_time
                    )
                    
                    if should_send_update:
                        logger.info("Updating position prices and holding times...")
                        for symbol, position in self.portfolio.positions.items():
                            try:
                                # Get current price
                                stock = yf.Ticker(symbol)
                                current_price = None
                                
                                # Try live data first
                                live_data = stock.history(period='1d', interval='1m')
                                if not live_data.empty:
                                    current_price = float(live_data['Close'].iloc[-1])
                                else:
                                    # Fallback to regular data
                                    info = stock.info
                                    current_price = float(
                                        info.get('regularMarketPrice') or 
                                        info.get('currentPrice') or 
                                        info.get('previousClose', 0)
                                    )
                                
                                if current_price and current_price > 0:
                                    # Update position
                                    position.current_price = current_price
                                    position.last_updated = datetime.now(timezone.utc)
                                    
                                    # Log significant price changes
                                    price_change = ((current_price - position.entry_price) / position.entry_price) * 100
                                    if abs(price_change) >= 2:  # Log changes >= 2%
                                        logger.info(
                                            f"Price update for {symbol}: ${current_price:.2f} "
                                            f"({price_change:+.2f}% from entry)"
                                        )
                                        
                                    # Send notification for significant changes
                                    if abs(price_change) >= 5:  # Notify on changes >= 5%
                                        await self.notifier.send_message(
                                            f"💰 <b>Significant Price Change for {symbol}</b>\n"
                                            f"Current: ${current_price:.2f}\n"
                                            f"Change: {price_change:+.2f}%\n"
                                            f"Holding Time: {position.holding_duration:.1f} days"
                                        )
                                
                            except Exception as e:
                                logger.error(f"Error updating position for {symbol}: {str(e)}")
                                continue
                        
                        # Send portfolio update at market transitions
                        summary_type = "Market Open" if is_market_open_time else "Market Close"
                        logger.info(f"Sending scheduled portfolio update at {summary_type}")
                        await self._notify_portfolio_update()
                        last_update_time = current_key
                        
            except Exception as e:
                logger.error(f"Error in position update loop: {str(e)}")
            
            # Update every minute during any trading session
            await asyncio.sleep(60)

    async def start(self, symbols: list[str]):
        """Start monitoring with web server for Telegram commands"""
        try:
            logger.info(f"Starting monitoring for {len(symbols)} symbols...")
            
            # Start web server for Telegram commands
            try:
                await self.start_web_server()
                logger.info("Web server started successfully")
            except Exception as e:
                logger.error(f"Failed to start web server: {str(e)}")
                logger.warning("Continuing without web server functionality")
            
            # Start monitoring tasks
            monitoring_tasks = [
                self.monitor_stock(symbol) for symbol in symbols
            ]
            # Add position update task
            monitoring_tasks.append(self.update_positions())
            
            logger.info("All monitoring tasks started")
            
            # Run all tasks
            await asyncio.gather(*monitoring_tasks)
            
        except Exception as e:
            logger.error(f"Error in monitoring tasks: {str(e)}")
            self.is_running = False

    async def get_position_summary(self) -> str:
        """Generate summary of all active positions"""
        summary = ["Active Positions:"]
        for position in self.active_positions.values():
            summary.append(
                f"\n{position.symbol}:\n"
                f"  Holding Duration: {position.holding_duration:.1f} days\n"
                f"  Entry Price: ${position.entry_price:.2f}\n"
                f"  Current Price: ${position.current_price:.2f}\n"
                f"  P/L: {position.profit_loss:.2f}%\n"
                f"  Sentiment Change: {position.sentiment_change:.2f}"
            )
        return "\n".join(summary) if self.active_positions else "No active positions"

    async def _notify_market_closed(self, message: str):
        """Send market closed notification"""
        try:
            await self.notifier.send_message(
                f"🏦 <b>Market Status Update</b>\n\n{message}"
            )
            logger.info("Sent market closed notification")
        except Exception as e:
            logger.error(f"Failed to send market closed notification: {str(e)}")

    async def _notify_market_state_change(self, is_open: bool):
        """Send notification when market state changes"""
        try:
            current_time = self.get_et_time()
            if is_open:
                message = (
                    f"🟢 <b>Market Now Open</b>\n"
                    f"Time: {current_time.strftime('%I:%M %p ET')}\n"
                    f"Starting regular monitoring cycle..."
                )
            else:
                message = (
                    f"🔴 <b>Market Now Closed</b>\n"
                    f"Time: {current_time.strftime('%I:%M %p ET')}\n"
                    f"Switching to 6-hour monitoring intervals..."
                )
            await self.notifier.send_message(message)
            logger.info(f"Market state change notification sent: {'Open' if is_open else 'Closed'}")
        except Exception as e:
            logger.error(f"Failed to send market state change notification: {str(e)}")

async def main():
    try:
        monitor = StockMonitor()
        
        # Get initial symbols
        symbols = get_all_symbols()
        if not symbols:
            logger.error("Failed to fetch stock symbols")
            return
            
        # Start monitor with symbols
        await monitor.start(symbols)
        
        processed_symbols = set()
        cycle_analyses = {}
        
        while monitor.is_running:
            try:
                current_time = monitor.get_et_time()
                is_market_open = monitor.is_market_open()
                
                # Check for market state change and notify if changed
                if monitor.last_market_state is None:
                    monitor.last_market_state = is_market_open
                elif monitor.last_market_state != is_market_open:
                    await monitor._notify_market_state_change(is_market_open)
                    monitor.last_market_state = is_market_open
                
                # Determine sleep duration based on market status
                if not is_market_open:
                    logger.info("Market is closed. Will check again in 6 hours...")
                    await asyncio.sleep(6 * 60 * 60)  # Sleep for 6 hours
                else:
                    # Regular market hours - continue with normal frequency
                    logger.info("Starting new monitoring cycle...")
                
                # Reset processed symbols for new cycle
                processed_symbols.clear()
                cycle_analyses.clear()
                
                # Process symbols in smaller batches with memory management
                batch_size = 50  # Increased batch size as requested
                total_batches = (len(symbols) + batch_size - 1) // batch_size
                
                for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, len(symbols))} of {len(symbols)} stocks)")
                    batch = symbols[i:i + batch_size]
                    batch_tasks = []
                    batch_analyses = {}
                    
                    # Progressive breaks based on letter ranges
                    if any(symbol[0] >= 'S' for symbol in batch):
                        logger.info("Reached stocks starting with S-Z. Taking an extended break...")
                        await asyncio.sleep(300)  # 5 minute break before S+ stocks
                    elif any(symbol[0] >= 'M' for symbol in batch):
                        logger.info("Reached stocks starting with M-R. Taking a longer break...")
                        await asyncio.sleep(360)  # 6 minute break before M+ stocks (increased from 4)
                    elif any(symbol[0] >= 'G' for symbol in batch):
                        logger.info("Reached stocks starting with G-L. Taking a medium break...")
                        await asyncio.sleep(180)   # 3 minute break before G+ stocks
                    
                    rate_limit_count = 0  # Track consecutive rate limits
                    
                    for symbol in batch:
                        if symbol not in processed_symbols:
                            processed_symbols.add(symbol)
                            
                            # Exponential backoff if we hit too many rate limits
                            if rate_limit_count > 0:
                                backoff_time = min(900, 60 * (2 ** (rate_limit_count - 1)))  # Max 15 minutes (increased from 10)
                                logger.info(f"Rate limit backoff: waiting {backoff_time} seconds...")
                                await asyncio.sleep(backoff_time)
                            
                            retry_count = 0
                            while retry_count < 3:  # Retry up to 3 times
                                try:
                                    # Get news and analyze stock
                                    news_articles = await monitor.news_service.fetch_stock_news(symbol)
                                    if news_articles:
                                        analysis = monitor.stock_analyzer.analyze_stock(symbol, news_articles)
                                        if analysis:
                                            batch_analyses[symbol] = analysis
                                            logger.debug(f"Added analysis for {symbol}")
                                    
                                    # Clear news articles to free memory
                                    del news_articles
                                    
                                    # Reset rate limit count on successful request
                                    rate_limit_count = 0
                                    
                                    break  # Exit retry loop on success
                                    
                                except Exception as e:
                                    if "429" in str(e):  # Rate limit error
                                        rate_limit_count += 1
                                        retry_count += 1
                                        logger.warning(f"Rate limit hit for {symbol} (count: {rate_limit_count})")
                                        
                                        # Aggressive backoff for M+ stocks
                                        if symbol[0] >= 'M':
                                            backoff_time = min(1800, 120 * (2 ** (rate_limit_count - 1)))  # Max 30 minutes
                                            logger.info(f"Aggressive rate limit backoff for {symbol}: waiting {backoff_time} seconds...")
                                            await asyncio.sleep(backoff_time)
                                        else:
                                            await asyncio.sleep(60)  # 1 minute backoff for other stocks
                                    else:
                                        logger.error(f"Error analyzing {symbol}: {str(e)}")
                                        break  # Exit retry loop on non-rate limit error
                            
                            else:
                                # Exhausted retries, move on to next symbol
                                logger.warning(f"Giving up on {symbol} after {retry_count} retries. Continuing...")
                            
                            # Progressive delays based on symbol
                            if symbol[0] >= 'S':
                                await asyncio.sleep(2.0)  # 2.0s delay for S-Z stocks (increased from 1.5)
                            elif symbol[0] >= 'M':
                                await asyncio.sleep(1.5)  # 1.5s delay for M-R stocks (increased from 1.0)
                            elif symbol[0] >= 'G':
                                await asyncio.sleep(1.0)  # 1.0s delay for G-L stocks (increased from 0.7)
                            else:
                                await asyncio.sleep(0.5)  # 0.5s delay for A-F stocks (increased from 0.3)
                    
                    if batch_tasks:
                        # Wait for batch to complete
                        await asyncio.gather(*batch_tasks)
                        
                        # Update cycle analyses with batch results
                        cycle_analyses.update(batch_analyses)
                        
                        # Clear batch data
                        batch_analyses.clear()
                        for task in batch_tasks:
                            task.cancel()
                        batch_tasks.clear()
                        
                        # Progressive delays between batches based on letter ranges
                        if any(symbol[0] >= 'S' for symbol in batch):
                            logger.info("S-Z batch completed. Taking extended break...")
                            await asyncio.sleep(90)  # 90s between S-Z batches (increased from 60)
                        elif any(symbol[0] >= 'M' for symbol in batch):
                            logger.info("M-R batch completed. Taking longer break...")
                            await asyncio.sleep(75)  # 75s between M-R batches (increased from 45)
                        elif any(symbol[0] >= 'G' for symbol in batch):
                            logger.info("G-L batch completed. Taking medium break...")
                            await asyncio.sleep(45)  # 45s between G-L batches (increased from 30)
                        else:
                            logger.info("A-F batch completed. Taking short break...")
                            await asyncio.sleep(30)  # 30s between A-F batches (increased from 15)
                        
                        # Force garbage collection after each batch
                        import gc
                        gc.collect()
                        
                        # Log progress with more details
                        logger.info(f"Completed batch {batch_num}/{total_batches} - Processed {len(batch)} stocks - Memory cleaned")
                
                logger.info(f"Analyzed {len(cycle_analyses)} stocks in detail")
                
                # If portfolio is empty, show a random stock analysis
                if not monitor.portfolio.positions and cycle_analyses:
                    # Get 3 random stocks that had analysis this cycle
                    sample_size = min(3, len(cycle_analyses))
                    random_symbols = random.sample(list(cycle_analyses.keys()), sample_size)
                    
                    for random_symbol in random_symbols:
                        analysis = cycle_analyses[random_symbol]
                        await monitor._notify_analysis_summary(analysis)
                
                # Clear analyses after processing
                cycle_analyses.clear()
                
                # Take a break before next cycle
                if monitor.is_market_open():
                    logger.info("Taking 2-minute break before starting next cycle...")
                    await asyncio.sleep(120)  # Regular cycle break during market hours
                else:
                    logger.info("Cycle complete during market closure. Next check in 6 hours.")
                    await asyncio.sleep(6 * 60 * 60)  # 6-hour break during market closure
                    
                # Force garbage collection between cycles
                gc.collect()
            
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {str(e)}")
                await asyncio.sleep(300)  # Sleep for 5 minutes on error
                continue
        
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}") 