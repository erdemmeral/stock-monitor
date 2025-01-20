import os
from dotenv import load_dotenv
import asyncio
import aiohttp

async def test_telegram():
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_IDS')
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message = "🔔 Test message from Stock Monitor Bot"
    
    async with aiohttp.ClientSession() as session:
        params = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        async with session.post(url, params=params) as response:
            result = await response.json()
            if response.status == 200:
                print("✅ Test message sent successfully!")
            else:
                print(f"❌ Error: {result}")

if __name__ == "__main__":
    asyncio.run(test_telegram()) 