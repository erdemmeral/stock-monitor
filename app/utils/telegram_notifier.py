import os
import aiohttp
import logging
from ..models.stock import StockAnalysis
from ..models.position import Position
from ..models.news import NewsArticle
from typing import Optional, List
import time
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        # Try both singular and plural environment variables for chat ID
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID') or os.getenv('TELEGRAM_CHAT_IDS')
        
        if not self.bot_token or not self.chat_id:
            logger.error(f"Telegram credentials missing: token={bool(self.bot_token)}, chat_id={bool(self.chat_id)}")
        else:
            logger.info("Telegram credentials loaded successfully")
            
        self.api_base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.commands = {
            '/help': self.cmd_help,
            '/status': self.cmd_status,
            '/portfolio': self.cmd_portfolio,
            '/active': self.cmd_active,
            '/performance': self.cmd_performance
        }
        self.monitor = None  # Will be set by the monitor

    def set_monitor(self, monitor):
        """Set the monitor instance for accessing portfolio data"""
        self.monitor = monitor

    async def handle_command(self, command: str) -> str:
        """Handle incoming commands"""
        cmd_func = self.commands.get(command.lower())
        if cmd_func:
            return await cmd_func()
        return "Unknown command. Type /help for available commands."

    async def cmd_help(self) -> str:
        """Handle /help command"""
        return (
            "📱 <b>Available Commands:</b>\n"
            "/help - Show this help message\n"
            "/status - Check system status\n"
            "/portfolio - View full portfolio summary\n"
            "/active - View active positions\n"
            "/performance - View performance metrics"
        )

    async def cmd_status(self) -> str:
        """Handle /status command"""
        if not self.monitor:
            return "System status unavailable"
        
        return (
            "🔄 <b>System Status</b>\n"
            f"Running: {self.monitor.is_running}\n"
            f"Last Update: {self.monitor.last_heartbeat.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Active Signals: {len(self.monitor.active_buy_signals)}\n"
            f"Update Interval: {self.monitor.update_interval} seconds"
        )

    async def cmd_portfolio(self) -> str:
        """Handle /portfolio command"""
        if not self.monitor or not self.monitor.portfolio:
            return "Portfolio data unavailable"
        return self.monitor.portfolio.get_portfolio_summary()

    async def cmd_active(self) -> str:
        """Handle /active command"""
        if not self.monitor or not self.monitor.portfolio:
            return "Portfolio data unavailable"
        
        positions = self.monitor.portfolio.positions
        if not positions:
            return "No active positions"
        
        summary = ["🟢 <b>Active Positions:</b>"]
        for pos in positions.values():
            summary.append(
                f"\n{pos.symbol}:"
                f"\n  Entry: ${pos.entry_price:.2f}"
                f"\n  Current: ${pos.current_price:.2f}"
                f"\n  P/L: {pos.profit_loss:.2f}%"
            )
        return "\n".join(summary)

    async def cmd_performance(self) -> str:
        """Handle /performance command"""
        if not self.monitor or not self.monitor.portfolio:
            return "Portfolio data unavailable"
        
        closed = self.monitor.portfolio.closed_positions
        if not closed:
            return "No closed positions yet"
        
        total_pl = sum(pos['profit_loss'] for pos in closed)
        avg_pl = total_pl / len(closed)
        win_trades = sum(1 for pos in closed if pos['profit_loss'] > 0)
        win_rate = (win_trades / len(closed)) * 100
        
        return (
            "📊 <b>Performance Metrics</b>\n"
            f"Total Trades: {len(closed)}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Average P/L: {avg_pl:.2f}%\n"
            f"Total P/L: {total_pl:.2f}%"
        )

    async def send_message(self, message: str):
        """Send message to Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials not configured - "
                        f"token: {'present' if self.bot_token else 'missing'}, "
                        f"chat_id: {'present' if self.chat_id else 'missing'}")
            return

        url = f"{self.api_base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        logger.error(f"Failed to send Telegram message: {error_data}")
                    else:
                        logger.debug(f"Successfully sent Telegram message: {message[:50]}...")
                    return await response.json()
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def _get_news_importance_indicator(self, title: str) -> str:
        """Get importance indicator emoji based on news title keywords"""
        title_lower = title.lower()
        
        high_impact_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'acquisition', 'merger', 'buyout', 'takeover', 'patent',
            'lawsuit', 'investigation', 'fda', 'approval', 'contract',
            'partnership', 'restructuring', 'layoff', 'ceo', 'executive'
        }
        
        medium_impact_keywords = {
            'market share', 'growth', 'expansion', 'investment', 'launch',
            'development', 'research', 'upgrade', 'downgrade', 'target',
            'analyst', 'recommendation', 'outlook', 'trend', 'performance'
        }
        
        if any(keyword in title_lower for keyword in high_impact_keywords):
            return "⚡️"  # High importance
        elif any(keyword in title_lower for keyword in medium_impact_keywords):
            return "📈"  # Medium importance
        return "📰"  # Regular news

    async def send_buy_signal(self, analysis: StockAnalysis, relevant_news: List[NewsArticle]):
        """Send buy signal notification with relevant news links"""
        # Format news links section
        news_section = "\n\n<b>Relevant News:</b>"
        for article in relevant_news:
            # Calculate how old the news is
            age_hours = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
            if age_hours < 24:
                time_ago = f"{int(age_hours)} hours ago"
            else:
                days = int(age_hours / 24)
                time_ago = f"{days} days ago"
            
            importance = self._get_news_importance_indicator(article.title)
            news_section += f"\n{importance} <a href='{article.url}'>{article.title}</a>"
            news_section += f"\n  ({article.source}, {time_ago}, Sentiment: {article.sentiment_score:.2f})"

        message = (
            f"🟢 <b>BUY SIGNAL: {analysis.symbol}</b>\n\n"
            f"Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Price: ${analysis.current_price:.2f}\n"
            f"Sentiment: {analysis.sentiment_score:.2f}\n"
            f"Confidence: {analysis.confidence_score:.2f}\n"
            f"Recommendation: {analysis.recommendation}"
            f"{news_section}\n\n"
            f"📊 News Importance:\n"
            f"⚡️ High Impact News\n"
            f"📈 Medium Impact News\n"
            f"📰 Regular News"
        )
        await self.send_message(message)

    async def send_sell_signal(self, analysis: StockAnalysis, prev_analysis: StockAnalysis, 
                             position: Optional[Position] = None, negative_news: List[NewsArticle] = None):
        """Send sell signal notification with position details and negative news"""
        position_details = ""
        if position:
            position_details = (
                f"\n\n<b>Position Summary:</b>\n"
                f"Holding Duration: {position.holding_duration:.1f} days\n"
                f"Entry Price: ${position.entry_price:.2f}\n"
                f"Total P/L: {position.profit_loss:.2f}%\n"
                f"Sentiment Change: {position.sentiment_change:.2f}"
            )

        # Add negative news section if available
        news_section = ""
        if negative_news:
            news_section = "\n\n<b>Negative News Triggering Sell:</b>"
            for article in negative_news:
                # Calculate how old the news is
                age_hours = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
                if age_hours < 24:
                    time_ago = f"{int(age_hours)} hours ago"
                else:
                    days = int(age_hours / 24)
                    time_ago = f"{days} days ago"
                
                importance = self._get_news_importance_indicator(article.title)
                news_section += f"\n{importance} <a href='{article.url}'>{article.title}</a>"
                news_section += f"\n  ({article.source}, {time_ago}, Sentiment: {article.sentiment_score:.2f})"

        message = (
            f"🔴 <b>SELL SIGNAL: {analysis.symbol}</b>\n\n"
            f"Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Current Price: ${analysis.current_price:.2f}\n"
            f"Previous Price: ${prev_analysis.current_price:.2f}\n"
            f"Current Sentiment: {analysis.sentiment_score:.2f}\n"
            f"Previous Sentiment: {prev_analysis.sentiment_score:.2f}\n"
            f"Confidence: {analysis.confidence_score:.2f}"
            f"{position_details}"
            f"{news_section}\n\n"
            f"📊 News Importance:\n"
            f"⚡️ High Impact News\n"
            f"📈 Medium Impact News\n"
            f"📰 Regular News"
        )
        await self.send_message(message)

    async def send_confidence_update(self, new_analysis: StockAnalysis, prev_analysis: StockAnalysis):
        """Send confidence increase notification"""
        message = (
            f"📈 <b>CONFIDENCE INCREASE: {new_analysis.symbol}</b>\n\n"
            f"Time: {new_analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Price: ${new_analysis.current_price:.2f}\n"
            f"New Confidence: {new_analysis.confidence_score:.2f}\n"
            f"Previous Confidence: {prev_analysis.confidence_score:.2f}"
        )
        await self.send_message(message)

    async def send_portfolio_update(self, portfolio_summary: str):
        """Send portfolio summary"""
        message = f"📊 <b>Portfolio Summary</b>\n{portfolio_summary}"
        await self.send_message(message) 