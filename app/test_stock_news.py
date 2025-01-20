import asyncio
from app.services.news_service import NewsService
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.stock_analyzer import StockAnalyzer
from app.utils.sp500 import get_sp500_symbols
from datetime import datetime, timezone

async def test_stock_analysis(symbol: str):
    """
    Test the complete flow of fetching news, analyzing sentiment,
    and generating stock recommendations
    """
    print(f"\nAnalyzing {symbol}...")
    
    # Initialize services
    news_service = NewsService()
    sentiment_analyzer = SentimentAnalyzer()
    stock_analyzer = StockAnalyzer()
    
    try:
        # Fetch news
        print("\nFetching news from the last hour...")
        news_articles = await news_service.fetch_stock_news(symbol)
        
        if not news_articles:
            print(f"No news articles found for {symbol} in the last hour")
            return
            
        print(f"Found {len(news_articles)} recent articles")
        
        # Analyze sentiment for each article
        print("\nAnalyzing sentiment...")
        now = datetime.now(timezone.utc)
        for article in news_articles:
            article.sentiment_score = sentiment_analyzer.analyze_sentiment(article)
            time_ago = now - article.published_at
            minutes_ago = int(time_ago.total_seconds() / 60)
            
            print(f"\nArticle: {article.title}")
            print(f"Source: {article.source}")
            print(f"Published: {minutes_ago} minutes ago")
            print(f"Sentiment Score: {article.sentiment_score:.2f}")
            print(f"URL: {article.url}")
        
        # Generate stock analysis
        print("\nGenerating stock analysis...")
        analysis = stock_analyzer.analyze_stock(symbol, news_articles)
        
        if analysis:
            print("\nAnalysis Results:")
            print("=" * 50)
            print(f"Symbol: {analysis.symbol}")
            print(f"Current Price: ${analysis.current_price:.2f}")
            print(f"Sentiment Score: {analysis.sentiment_score:.2f}")
            print(f"News Impact Score: {analysis.news_impact_score:.2f}")
            print(f"Confidence Score: {analysis.confidence_score:.2f}")
            print(f"Recommendation: {analysis.recommendation}")
            print(f"Should Buy: {analysis.should_buy()}")
            print(f"Analysis Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print("=" * 50)
        else:
            print("\nAnalysis failed - Could not generate recommendation")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")

def get_stock_symbol():
    while True:
        print("\nOptions:")
        print("1. Analyze single stock")
        print("2. Analyze all S&P 500 stocks")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            symbol = input("Enter stock ticker symbol (e.g., AAPL, GOOGL): ").strip().upper()
            if symbol:
                return ("single", symbol)
            print("Please enter a valid stock symbol.")
            
        elif choice == "2":
            return ("sp500", None)
            
        elif choice == "3":
            return ("quit", None)
            
        else:
            print("Invalid choice. Please try again.")

async def analyze_sp500():
    """Analyze all S&P 500 stocks and show top recommendations"""
    print("\nFetching S&P 500 symbols...")
    symbols = get_sp500_symbols()
    
    if not symbols:
        print("Failed to fetch S&P 500 symbols.")
        return
        
    print(f"Analyzing {len(symbols)} S&P 500 stocks...")
    
    results = []
    total_analyzed = 0
    stocks_with_news = 0
    
    for symbol in symbols:
        try:
            print(f"\nAnalyzing {symbol}...", end='', flush=True)
            news_service = NewsService()
            sentiment_analyzer = SentimentAnalyzer()
            stock_analyzer = StockAnalyzer()
            
            news_articles = await news_service.fetch_stock_news(symbol)
            total_analyzed += 1
            
            if news_articles:
                stocks_with_news += 1
                print(f" Found {len(news_articles)} articles")
                
                # Analyze sentiment for each article
                for article in news_articles:
                    article.sentiment_score = sentiment_analyzer.analyze_sentiment(article)
                    
                # Generate stock analysis
                analysis = stock_analyzer.analyze_stock(symbol, news_articles)
                if analysis and analysis.should_buy():  # Only add if it's a buy recommendation
                    results.append(analysis)
                    print(f" -> {analysis.recommendation} recommendation (Confidence: {analysis.confidence_score:.2f})")
            else:
                print(" No recent news")
                
        except Exception as e:
            print(f"\nError analyzing {symbol}: {str(e)}")
            continue
        
        # Add a small delay to avoid rate limiting
        await asyncio.sleep(0.5)
    
    # Sort and display results
    if results:
        print("\nAnalysis Summary:")
        print(f"Total stocks analyzed: {total_analyzed}")
        print(f"Stocks with news: {stocks_with_news}")
        print(f"Stocks with buy recommendations: {len(results)}")
        
        print("\nTop S&P 500 Stock Recommendations:")
        print("=" * 50)
        sorted_results = sorted(results, key=lambda x: x.confidence_score, reverse=True)
        for analysis in sorted_results[:10]:  # Show top 10
            print(f"\nSymbol: {analysis.symbol}")
            print(f"Current Price: ${analysis.current_price:.2f}")
            print(f"Sentiment Score: {analysis.sentiment_score:.2f}")
            print(f"News Impact Score: {analysis.news_impact_score:.2f}")
            print(f"Confidence Score: {analysis.confidence_score:.2f}")
            print(f"Recommendation: {analysis.recommendation}")
            print("-" * 30)
    else:
        print("\nNo buy recommendations found. Analysis Summary:")
        print(f"Total stocks analyzed: {total_analyzed}")
        print(f"Stocks with news: {stocks_with_news}")

if __name__ == "__main__":
    print("Welcome to Stock News Analyzer!")
    print("This program analyzes news sentiment and makes stock recommendations.")
    
    while True:
        mode, symbol = get_stock_symbol()
        
        if mode == "quit":
            print("\nThank you for using Stock News Analyzer!")
            break
            
        try:
            if mode == "single":
                asyncio.run(test_stock_analysis(symbol))
            elif mode == "sp500":
                asyncio.run(analyze_sp500())
                
        except Exception as e:
            print(f"\nError in analysis: {str(e)}")
        
        print("\n" + "="*50)  # Separator between analyses 