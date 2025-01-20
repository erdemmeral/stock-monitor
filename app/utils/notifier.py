import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ..models.stock import StockAnalysis

class EmailNotifier:
    def __init__(self):
        self.sender_email = os.getenv('EMAIL_ADDRESS')
        self.sender_password = os.getenv('EMAIL_PASSWORD')  # App password for Gmail
        self.recipient_email = os.getenv('RECIPIENT_EMAIL')
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_notification(self, subject: str, body: str):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

        except Exception as e:
            print(f"Failed to send email: {str(e)}")

    def send_buy_signal(self, analysis: StockAnalysis):
        subject = f"BUY SIGNAL: {analysis.symbol}"
        body = (
            f"BUY SIGNAL DETECTED\n"
            f"Symbol: {analysis.symbol}\n"
            f"Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Current Price: ${analysis.current_price:.2f}\n"
            f"Sentiment Score: {analysis.sentiment_score:.2f}\n"
            f"Confidence Score: {analysis.confidence_score:.2f}\n"
            f"Recommendation: {analysis.recommendation}\n"
        )
        self.send_notification(subject, body) 

    def send_sell_signal(self, analysis: StockAnalysis, prev_analysis: StockAnalysis):
        subject = f"SELL SIGNAL: {analysis.symbol}"
        body = (
            f"SELL SIGNAL DETECTED\n"
            f"Symbol: {analysis.symbol}\n"
            f"Time: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Current Price: ${analysis.current_price:.2f}\n"
            f"Previous Price: ${prev_analysis.current_price:.2f}\n"
            f"Price Change: {((analysis.current_price - prev_analysis.current_price) / prev_analysis.current_price * 100):.2f}%\n"
            f"Current Sentiment: {analysis.sentiment_score:.2f}\n"
            f"Previous Sentiment: {prev_analysis.sentiment_score:.2f}\n"
            f"Confidence Score: {analysis.confidence_score:.2f}\n"
            f"Recommendation: SELL\n"
        )
        self.send_notification(subject, body) 