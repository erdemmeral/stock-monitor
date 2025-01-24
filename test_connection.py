import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    url = "https://portfolio-tracker-rough-dawn-5271.fly.dev/api/positions"
    data = {
        'symbol': 'TEST',
        'entryPrice': 100.0,
        'targetPrice': 110.0,
        'entryDate': '2024-01-23T00:00:00.000Z',
        'targetDate': '2024-02-23T00:00:00.000Z'
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    try:
        async with aiohttp.ClientSession() as session:
            # First try a GET request to check if service is up
            logger.info("Testing GET request to base URL...")
            async with session.get("https://portfolio-tracker-rough-dawn-5271.fly.dev/health", ssl=False) as response:
                logger.info(f"GET Status: {response.status}")
                logger.info(f"GET Response: {await response.text()}")

            # Then try the POST request
            logger.info("\nTesting POST request...")
            logger.info(f"URL: {url}")
            logger.info(f"Data: {data}")
            async with session.post(url, json=data, headers=headers, ssl=False) as response:
                logger.info(f"POST Status: {response.status}")
                logger.info(f"POST Response: {await response.text()}")

    except Exception as e:
        logger.error(f"Connection error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_connection()) 