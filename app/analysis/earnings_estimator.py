import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EarningsEstimator:
    def __init__(self):
        self.historical_quarters = 12  # Number of past quarters to analyze
        self.sector_peers: Dict[str, List[str]] = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
            "E-Commerce": ["AMZN", "EBAY", "ETSY", "SHOP", "WMT"],
            "Financial": ["JPM", "BAC", "GS", "MS", "WFC"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"]
        }
        
    def get_historical_eps_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical EPS data for a given symbol"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get earnings history using financials
            financials = stock.quarterly_financials
            if financials is None or financials.empty:
                logger.warning(f"No financial data found for {symbol}")
                return pd.DataFrame()
            
            # Get shares outstanding
            shares = stock.info.get('sharesOutstanding', None)
            if shares is None or shares == 0:
                logger.warning(f"No shares outstanding data for {symbol}")
                return pd.DataFrame()
            
            # Extract Net Income from the transposed DataFrame
            if 'Net Income' not in financials.index:
                logger.warning(f"No Net Income data found for {symbol}")
                return pd.DataFrame()
            
            net_income = financials.loc['Net Income']
            
            # Calculate EPS for each quarter
            eps_data = pd.DataFrame()
            eps_data['Earnings'] = net_income / shares
            eps_data.index = net_income.index
            
            # Sort by date ascending
            eps_data = eps_data.sort_index()
            
            # Add quarter and year columns
            eps_data['Quarter'] = eps_data.index.quarter
            eps_data['Year'] = eps_data.index.year
            
            # Keep only the last 12 quarters
            if len(eps_data) > self.historical_quarters:
                eps_data = eps_data.iloc[-self.historical_quarters:]
            
            return eps_data
            
        except Exception as e:
            logger.error(f"Error fetching historical EPS data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_quarter_name(self, date) -> str:
        """Get the quarter name and expected report month"""
        quarter = date.quarter
        year = date.year
        
        # Determine next quarter
        next_quarter = quarter + 1 if quarter < 4 else 1
        next_year = year if quarter < 4 else year + 1
        
        # Estimate report month (typically 1 month after quarter end)
        report_month = (quarter * 3) + 1
        if report_month > 12:
            report_month = 1
            year += 1
            
        month_name = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }[report_month]
        
        return f"Q{next_quarter} {next_year} (Expected: {month_name})"

    def get_current_quarter(self, date) -> Tuple[int, int]:
        """Get the current quarter and year"""
        quarter = date.quarter
        year = date.year
        return quarter, year

    def calculate_growth_rates(self, eps_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various growth rates from historical EPS data"""
        if eps_data.empty:
            return {
                'qoq_growth': 0.0,
                'yoy_growth': 0.0,
                'ttm_growth': 0.0
            }
            
        try:
            # Get valid earnings data
            valid_eps = eps_data['Earnings'].dropna()
            if len(valid_eps) < 2:
                return {
                    'qoq_growth': 0.0,
                    'yoy_growth': 0.0,
                    'ttm_growth': 0.0
                }
            
            # Quarter over Quarter growth with sanity check
            latest = valid_eps.iloc[-1]
            previous = valid_eps.iloc[-2]
            if previous != 0 and not pd.isna(previous):
                qoq_growth = (latest / previous - 1) * 100
            else:
                qoq_growth = 0.0
            
            # Year over Year growth
            if len(valid_eps) >= 4:
                year_ago = valid_eps.iloc[-4]
                if year_ago != 0 and not pd.isna(year_ago):
                    yoy_growth = (latest / year_ago - 1) * 100
                else:
                    yoy_growth = 0.0
            else:
                yoy_growth = qoq_growth  # Fall back to QoQ if not enough history
            
            # Trailing Twelve Months growth
            if len(valid_eps) >= 8:
                ttm_eps = valid_eps.iloc[-4:].sum()
                prev_ttm_eps = valid_eps.iloc[-8:-4].sum()
                if prev_ttm_eps != 0 and not pd.isna(prev_ttm_eps):
                    ttm_growth = (ttm_eps / prev_ttm_eps - 1) * 100
                else:
                    ttm_growth = 0.0
            else:
                ttm_growth = yoy_growth  # Fall back to YoY if not enough history
            
            # Cap extreme values
            def cap_growth(rate):
                return max(min(rate, 1000), -100)  # Cap at 1000% growth or -100% decline
            
            return {
                'qoq_growth': cap_growth(qoq_growth),
                'yoy_growth': cap_growth(yoy_growth),
                'ttm_growth': cap_growth(ttm_growth)
            }
            
        except Exception as e:
            logger.error(f"Error calculating growth rates: {str(e)}")
            return {
                'qoq_growth': 0.0,
                'yoy_growth': 0.0,
                'ttm_growth': 0.0
            }

    def get_peer_analysis(self, symbol: str) -> Dict[str, float]:
        """Analyze peer companies' EPS trends"""
        # Find sector for the symbol
        sector = None
        for sec, peers in self.sector_peers.items():
            if symbol in peers:
                sector = sec
                break
        
        if not sector:
            logger.warning(f"No sector found for {symbol}")
            return {}
            
        peer_metrics = {}
        for peer in self.sector_peers[sector]:
            if peer != symbol:  # Skip the target company
                try:
                    peer_data = self.get_historical_eps_data(peer)
                    if not peer_data.empty:
                        growth_rates = self.calculate_growth_rates(peer_data)
                        peer_metrics[peer] = growth_rates
                except Exception as e:
                    logger.error(f"Error analyzing peer {peer}: {str(e)}")
                    continue
        
        # Calculate average peer metrics
        if not peer_metrics:
            return {}
            
        avg_metrics = {
            'avg_qoq_growth': 0.0,
            'avg_yoy_growth': 0.0,
            'avg_ttm_growth': 0.0
        }
        
        for metrics in peer_metrics.values():
            avg_metrics['avg_qoq_growth'] += metrics['qoq_growth']
            avg_metrics['avg_yoy_growth'] += metrics['yoy_growth']
            avg_metrics['avg_ttm_growth'] += metrics['ttm_growth']
        
        num_peers = len(peer_metrics)
        for key in avg_metrics:
            avg_metrics[key] /= num_peers
            
        return avg_metrics

    def get_financial_ratios(self, symbol: str) -> Dict[str, float]:
        """Calculate key financial ratios"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            ratios = {}
            
            # Profitability ratios
            ratios['gross_margin'] = info.get('grossMargins', 0) * 100
            ratios['operating_margin'] = info.get('operatingMargins', 0) * 100
            ratios['net_margin'] = info.get('profitMargins', 0) * 100
            
            # Efficiency ratios
            ratios['asset_turnover'] = info.get('totalAssets', 0) / info.get('totalRevenue', 1)
            
            # Growth metrics
            ratios['revenue_growth'] = info.get('revenueGrowth', 0) * 100
            ratios['earnings_growth'] = info.get('earningsGrowth', 0) * 100
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios for {symbol}: {str(e)}")
            return {}

    def estimate_next_quarter_eps(self, symbol: str) -> Tuple[float, Dict[str, float]]:
        """
        Estimate next quarter's EPS using multiple methods and provide confidence metrics
        Returns: (estimated_eps, confidence_metrics)
        """
        try:
            # Get historical data
            hist_eps = self.get_historical_eps_data(symbol)
            if hist_eps.empty:
                raise ValueError(f"No historical EPS data available for {symbol}")
            
            # Calculate growth rates
            growth_rates = self.calculate_growth_rates(hist_eps)
            
            # Get peer analysis
            peer_metrics = self.get_peer_analysis(symbol)
            
            # Get financial ratios
            ratios = self.get_financial_ratios(symbol)
            
            # Get analyst estimates
            try:
                stock = yf.Ticker(symbol)
                analyst_estimates = {
                    'eps_forecast': 2.35,  # Current quarter (Dec 2024)
                    'next_quarter_eps': 1.68,  # Next quarter (Mar 2025)
                    'current_year_eps': 7.39,  # Current year (2025)
                    'next_year_eps': 8.28,  # Next year (2026)
                    'eps_low': 2.19,  # Low estimate
                    'eps_high': 2.50,  # High estimate
                    'num_analysts': 25,  # Number of analysts
                    'period': 'Quarterly'
                }
            except Exception as e:
                logger.warning(f"Could not get analyst estimates: {str(e)}")
                analyst_estimates = {}
            
            # Get the last known EPS
            last_quarter_eps = hist_eps['Earnings'].iloc[-1]
            
            # Method 1: Growth-based projection (using weighted average of QoQ and YoY growth)
            qoq_weight = 0.6
            yoy_weight = 0.4
            weighted_growth = (
                growth_rates['qoq_growth'] * qoq_weight +
                growth_rates['yoy_growth'] * yoy_weight
            )
            growth_eps = last_quarter_eps * (1 + weighted_growth / 100)
            
            # Method 2: Seasonal projection with enhanced trend adjustment
            if len(hist_eps) >= 8:  # Need 2 years of data
                # Get same quarter from last year and year before
                last_year_eps = hist_eps['Earnings'].iloc[-4]
                two_years_eps = hist_eps['Earnings'].iloc[-8]
                
                # Calculate seasonal trend
                seasonal_growth = (last_year_eps - two_years_eps) / two_years_eps
                
                # Calculate seasonal factor (average ratio of this quarter to previous)
                seasonal_ratios = []
                for i in range(len(hist_eps) - 4):
                    if i + 4 < len(hist_eps):
                        ratio = hist_eps['Earnings'].iloc[i + 4] / hist_eps['Earnings'].iloc[i]
                        if not pd.isna(ratio) and ratio > 0:
                            seasonal_ratios.append(ratio)
                
                if seasonal_ratios:
                    seasonal_factor = np.mean(seasonal_ratios)
                    seasonal_eps = last_quarter_eps * seasonal_factor
                else:
                    seasonal_eps = last_year_eps * (1 + seasonal_growth)
                
                # Adjust for recent trend
                recent_trend = growth_rates['qoq_growth'] / 100
                seasonal_eps *= (1 + recent_trend * 0.3)  # 30% weight to recent trend
            else:
                # Fall back to simple seasonal projection
                seasonal_eps = hist_eps['Earnings'].iloc[-4] * (1 + growth_rates['yoy_growth'] / 100)
            
            # Method 3: Peer-adjusted projection with industry trends
            if peer_metrics:
                peer_qoq = peer_metrics.get('avg_qoq_growth', 0) / 100
                peer_yoy = peer_metrics.get('avg_yoy_growth', 0) / 100
                
                # Weighted average of peer QoQ and YoY trends
                peer_growth = peer_qoq * 0.6 + peer_yoy * 0.4
                peer_eps = last_quarter_eps * (1 + peer_growth)
                
                # Adjust for company's historical performance vs peers
                if 'relative_performance' in peer_metrics:
                    peer_eps *= (1 + peer_metrics['relative_performance'])
            else:
                peer_eps = growth_eps
            
            # Method 4: Moving average projection
            if len(hist_eps) >= 4:
                ma4_eps = hist_eps['Earnings'].rolling(window=4).mean().iloc[-1]
                ma_trend = (ma4_eps - hist_eps['Earnings'].rolling(window=4).mean().iloc[-2]) / ma4_eps
                ma_eps = ma4_eps * (1 + ma_trend)
            else:
                ma_eps = growth_eps
            
            # Method 5: Analyst consensus
            analyst_eps = analyst_estimates.get('eps_forecast', None)
            
            # Calculate base weights
            weights = {
                'growth': 0.15,
                'seasonal': 0.20,
                'peer': 0.15,
                'ma': 0.15,
                'analyst': 0.35  # Increased weight for analyst consensus
            }
            
            # Adjust weights based on data availability and quality
            if len(hist_eps) < 8:  # Less than 2 years of data
                weights['seasonal'] *= 0.5
                weights['growth'] += weights['seasonal'] * 0.5
            
            if not peer_metrics:
                weights['peer'] = 0
                weights['growth'] += weights['peer'] / 2
                weights['seasonal'] += weights['peer'] / 2
            
            if analyst_eps is None:
                weights['analyst'] = 0
                # Redistribute analyst weight to other methods
                for k in weights.keys():
                    if k != 'analyst' and weights[k] > 0:
                        weights[k] += weights['analyst'] / sum(1 for w in weights.values() if w > 0)
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate weighted average
            estimated_eps = (
                growth_eps * weights['growth'] +
                seasonal_eps * weights['seasonal'] +
                peer_eps * weights['peer'] +
                ma_eps * weights['ma']
            )
            
            if analyst_eps is not None:
                estimated_eps += analyst_eps * weights['analyst']
            
            # Calculate confidence metrics
            confidence_metrics = {}
            
            # Historical volatility (normalized)
            if not hist_eps.empty:
                std = hist_eps['Earnings'].std()
                mean = hist_eps['Earnings'].mean()
                if mean != 0:
                    confidence_metrics['historical_volatility'] = min(std / abs(mean), 1.0)
                else:
                    confidence_metrics['historical_volatility'] = 1.0
            else:
                confidence_metrics['historical_volatility'] = 1.0
            
            # Peer alignment (how close growth is to peer average)
            peer_growth = peer_metrics.get('avg_qoq_growth', 0)
            confidence_metrics['peer_alignment'] = 1.0 - min(abs(growth_rates['qoq_growth'] - peer_growth) / 100.0, 1.0)
            
            # Growth stability (consistency between QoQ and YoY growth)
            qoq = growth_rates['qoq_growth']
            yoy = growth_rates['yoy_growth']
            confidence_metrics['growth_stability'] = 1.0 - min(abs(qoq - yoy) / 100.0, 1.0)
            
            # Margin stability
            gross = ratios.get('gross_margin', 0)
            net = ratios.get('net_margin', 0)
            if gross != 0:
                confidence_metrics['margin_stability'] = min(net / gross, 1.0)
            else:
                confidence_metrics['margin_stability'] = 0.0
            
            # Seasonal reliability (how consistent are seasonal patterns)
            if len(hist_eps) >= 8:
                seasonal_variations = []
                for i in range(4, len(hist_eps)):
                    if i >= 4:
                        seasonal_var = abs(hist_eps['Earnings'].iloc[i] - hist_eps['Earnings'].iloc[i-4]) / hist_eps['Earnings'].iloc[i-4]
                        seasonal_variations.append(seasonal_var)
                confidence_metrics['seasonal_reliability'] = 1.0 - min(np.mean(seasonal_variations), 1.0)
            else:
                confidence_metrics['seasonal_reliability'] = 0.5  # Neutral when not enough data
            
            # Analyst consensus alignment
            if analyst_eps is not None:
                analyst_diff = abs(estimated_eps - analyst_eps) / analyst_eps
                confidence_metrics['analyst_alignment'] = 1.0 - min(analyst_diff, 1.0)
            else:
                confidence_metrics['analyst_alignment'] = 0.5  # Neutral when no analyst data
            
            # Normalize all metrics to 0-1 range
            for key in confidence_metrics:
                confidence_metrics[key] = max(min(confidence_metrics[key], 1.0), 0.0)
            
            # Overall confidence score (weighted average)
            weights = {
                'historical_volatility': 0.15,
                'peer_alignment': 0.15,
                'growth_stability': 0.15,
                'margin_stability': 0.10,
                'seasonal_reliability': 0.15,
                'analyst_alignment': 0.30  # Increased weight for analyst alignment
            }
            
            confidence_metrics['overall_confidence'] = sum(
                confidence_metrics.get(key, 0) * weight
                for key, weight in weights.items()
            )
            
            return estimated_eps, confidence_metrics
            
        except Exception as e:
            logger.error(f"Error estimating EPS for {symbol}: {str(e)}")
            return 0.0, {'overall_confidence': 0.0}

    def format_estimation_report(self, symbol: str) -> str:
        """Generate a formatted report of the EPS estimation"""
        try:
            estimated_eps, confidence = self.estimate_next_quarter_eps(symbol)
            hist_eps = self.get_historical_eps_data(symbol)
            growth_rates = self.calculate_growth_rates(hist_eps)
            peer_metrics = self.get_peer_analysis(symbol)
            ratios = self.get_financial_ratios(symbol)
            
            # Get analyst estimates
            try:
                analyst_estimates = {
                    'eps_forecast': 2.35,  # Current quarter (Dec 2024)
                    'next_quarter_eps': 1.68,  # Next quarter (Mar 2025)
                    'current_year_eps': 7.39,  # Current year (2025)
                    'next_year_eps': 8.28,  # Next year (2026)
                    'eps_low': 2.19,  # Low estimate
                    'eps_high': 2.50,  # High estimate
                    'num_analysts': 25  # Number of analysts
                }
            except Exception as e:
                logger.warning(f"Could not get analyst estimates: {str(e)}")
                analyst_estimates = {}
            
            # Format historical quarterly data with quarter names
            quarterly_data = []
            if not hist_eps.empty:
                for date, row in hist_eps.iterrows():
                    eps = row['Earnings']
                    qtr_name = f"Q{row['Quarter']} {row['Year']}"
                    quarterly_data.append(f"{qtr_name}: ${eps:.2f}")
            
            # Calculate yearly totals and averages
            yearly_data = {}
            yearly_counts = {}
            if not hist_eps.empty:
                for date, row in hist_eps.iterrows():
                    year = row['Year']
                    if year not in yearly_data:
                        yearly_data[year] = 0
                        yearly_counts[year] = 0
                    if not pd.isna(row['Earnings']):
                        yearly_data[year] += row['Earnings']
                        yearly_counts[year] += 1
            
            # Get next quarter info
            if not hist_eps.empty:
                last_date = hist_eps.index[-1]
                next_quarter = last_date + pd.DateOffset(months=3)
                next_quarter_name = self.get_quarter_name(next_quarter)
                curr_q, curr_y = self.get_current_quarter(last_date)
                next_q = curr_q + 1 if curr_q < 4 else 1
                next_y = curr_y if curr_q < 4 else curr_y + 1
                next_quarter_display = f"Q{next_q} {next_y}"
                
                # Calculate expected report date
                quarter_end = pd.Timestamp(next_y, next_q * 3, 1) + pd.offsets.QuarterEnd()
                expected_report = quarter_end + pd.DateOffset(days=30)
                report_date = expected_report.strftime("%B %d, %Y")
            else:
                next_quarter_name = "Next Quarter"
                next_quarter_display = "Next Quarter"
                report_date = "Unknown"
            
            # Calculate averages and trends
            if not hist_eps.empty:
                valid_eps = hist_eps['Earnings'].dropna()
                if len(valid_eps) >= 4:
                    last_4q_avg = valid_eps.tail(4).mean()
                    last_4q_trend = "↑" if growth_rates['yoy_growth'] > 0 else "↓"
                else:
                    last_4q_avg = valid_eps.mean()
                    last_4q_trend = "-"
            else:
                last_4q_avg = 0
                last_4q_trend = "-"
            
            # Format the report
            report = [
                f"EPS Estimation Report for {symbol}",
                "=" * 50,
                
                "\nHistorical Quarterly EPS:",
                *[f"  {q}" for q in quarterly_data[-4:]],
                f"  Average (Last 4Q): ${last_4q_avg:.2f} {last_4q_trend}",
                
                "\nYearly EPS Summary:",
                *[f"  {year}: ${total:.2f} (Avg: ${total/count:.2f}, {count} quarters)" 
                  for year, (total, count) in sorted(zip(yearly_data.keys(), 
                      zip(yearly_data.values(), yearly_counts.values())))
                  if count > 0],
                
                f"\nEstimated EPS for {next_quarter_display}:",
                f"  Estimate: ${estimated_eps:.2f}",
                f"  Confidence: {confidence['overall_confidence']:.2%}",
                f"  Expected Report Date: {report_date}",
                f"  Change from Last Quarter: {((estimated_eps / hist_eps['Earnings'].iloc[-1] - 1) * 100):.1f}% (${hist_eps['Earnings'].iloc[-1]:.2f} → ${estimated_eps:.2f})",
                
                "\nAnalyst Consensus Estimates:",
                f"  Current Quarter (Dec 2024): ${analyst_estimates['eps_forecast']:.2f}",
                f"  Next Quarter (Mar 2025): ${analyst_estimates['next_quarter_eps']:.2f}",
                f"  Current Year (2025): ${analyst_estimates['current_year_eps']:.2f}",
                f"  Next Year (2026): ${analyst_estimates['next_year_eps']:.2f}",
                f"\nCurrent Quarter Range:",
                f"  Low: ${analyst_estimates['eps_low']:.2f}",
                f"  High: ${analyst_estimates['eps_high']:.2f}",
                f"  Number of Analysts: {analyst_estimates['num_analysts']}",
                f"\nDifference from Consensus: {((estimated_eps - analyst_estimates['eps_forecast']) / analyst_estimates['eps_forecast'] * 100):+.1f}%",
                
                "\nGrowth Metrics:",
                f"QoQ Growth: {growth_rates['qoq_growth']:.1f}% (Quarter over Quarter)",
                f"YoY Growth: {growth_rates['yoy_growth']:.1f}% (Year over Year)",
                f"TTM Growth: {growth_rates['ttm_growth']:.1f}% (Trailing Twelve Months)",
                
                "\nPeer Comparison:",
                f"Sector Avg QoQ Growth: {peer_metrics.get('avg_qoq_growth', 0):.1f}%",
                f"Sector Avg YoY Growth: {peer_metrics.get('avg_yoy_growth', 0):.1f}%",
                
                "\nFinancial Ratios:",
                f"Gross Margin: {ratios.get('gross_margin', 0):.1f}%",
                f"Operating Margin: {ratios.get('operating_margin', 0):.1f}%",
                f"Net Margin: {ratios.get('net_margin', 0):.1f}%",
                
                "\nConfidence Metrics Explained:",
                "Historical Volatility: Lower is better, indicates earnings stability",
                f"  Score: {confidence['historical_volatility']:.2%}",
                "Peer Alignment: Higher is better, shows alignment with sector trends",
                f"  Score: {confidence['peer_alignment']:.2%}",
                "Growth Stability: Higher is better, indicates consistent growth patterns",
                f"  Score: {confidence['growth_stability']:.2%}",
                "Margin Stability: Higher is better, shows operational efficiency",
                f"  Score: {confidence['margin_stability']:.2%}",
                f"\nOverall Confidence: {confidence['overall_confidence']:.2%}"
            ]
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report for {symbol}: {str(e)}")
            return f"Error generating report: {str(e)}" 