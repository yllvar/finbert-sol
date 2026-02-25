"""
FinBERT Sentiment Analysis for SOL Trading System
Implements the paper's sentiment analysis methodology
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis for financial news
    Paper methodology: sentiment scores as filters, not signals
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        
        print(f"🤖 Initializing FinBERT on {self.device}...")
        self._load_model()
        print("✅ FinBERT loaded successfully")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Also load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = self.model.cuda()
                
        except Exception as e:
            print(f"⚠️  Error loading FinBERT: {e}")
            print("   Using mock sentiment analysis for development")
            self.sentiment_pipeline = None
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if self.sentiment_pipeline is None:
            # Mock sentiment for development
            return self._mock_sentiment(text)
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            # Get sentiment scores
            results = self.sentiment_pipeline(text)
            
            # Process results
            sentiment_scores = {}
            for result in results[0]:  # results is a list containing one dict
                sentiment_scores[result['label'].lower()] = result['score']
            
            # Calculate normalized sentiment score (-1 to 1)
            positive = sentiment_scores.get('positive', 0)
            negative = sentiment_scores.get('negative', 0)
            neutral = sentiment_scores.get('neutral', 0)
            
            # Normalize to -1 to 1 scale
            normalized_score = (positive - negative)
            
            return {
                'label': results[0][0]['label'],  # Most likely sentiment
                'confidence': results[0][0]['score'],
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'normalized_score': normalized_score,
                'text_length': len(text)
            }
            
        except Exception as e:
            print(f"⚠️  Error analyzing sentiment: {e}")
            return self._mock_sentiment(text)
    
    def _mock_sentiment(self, text: str) -> Dict:
        """Mock sentiment analysis for development/testing"""
        # Simple keyword-based mock sentiment
        positive_keywords = ['good', 'great', 'excellent', 'positive', 'up', 'bullish', 'growth', 'rally']
        negative_keywords = ['bad', 'terrible', 'negative', 'down', 'bearish', 'decline', 'crash', 'fall']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            label = 'positive'
            confidence = min(0.9, 0.5 + positive_count * 0.1)
        elif negative_count > positive_count:
            label = 'negative'
            confidence = min(0.9, 0.5 + negative_count * 0.1)
        else:
            label = 'neutral'
            confidence = 0.5
        
        normalized_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        return {
            'label': label,
            'confidence': confidence,
            'positive': confidence if label == 'positive' else 0.1,
            'negative': confidence if label == 'negative' else 0.1,
            'neutral': confidence if label == 'neutral' else 0.1,
            'normalized_score': normalized_score,
            'text_length': len(text),
            'mock': True
        }
    
    def analyze_batch_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        print(f"📊 Analyzing sentiment for {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(texts)}")
            
            result = self.analyze_text_sentiment(text)
            result['index'] = i
            results.append(result)
        
        print(f"✅ Batch analysis complete")
        return results
    
    def aggregate_daily_sentiment(self, news_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Aggregate sentiment scores by day (paper's methodology)
        
        Args:
            news_data: Dictionary mapping dates to lists of news articles
        
        Returns:
            Dictionary mapping dates to sentiment scores
        """
        daily_sentiment = {}
        
        print(f"📅 Aggregating daily sentiment for {len(news_data)} days...")
        
        for date, articles in news_data.items():
            if not articles:
                daily_sentiment[date] = 0.0
                continue
            
            # Analyze each article
            daily_scores = []
            valid_articles = 0
            
            for article in articles:
                # Only consider news before 9:30 AM EST (paper's rule)
                if 'time' in article and article['time'] < time(9, 30):
                    sentiment = self.analyze_text_sentiment(article['text'])
                    # Convert to -1 to 1 scale
                    score = sentiment['normalized_score']
                    daily_scores.append(score)
                    valid_articles += 1
            
            # Aggregate daily sentiment (paper's formula)
            if daily_scores:
                daily_sentiment[date] = np.mean(daily_scores)
            else:
                daily_sentiment[date] = 0.0
            
            print(f"   {date}: {len(daily_scores)} articles, sentiment={daily_sentiment[date]:.3f}")
        
        return daily_sentiment
    
    def create_sentiment_filter(self, sentiment_scores: Dict[str, float], 
                              threshold: float = -0.70) -> Dict[str, bool]:
        """
        Create sentiment filter as per paper methodology
        
        Args:
            sentiment_scores: Daily sentiment scores
            threshold: Sentiment threshold (paper uses -0.70)
        
        Returns:
            Dictionary mapping dates to boolean filter values
        """
        print(f"🎯 Creating sentiment filter with threshold {threshold}")
        
        sentiment_filter = {}
        blocked_days = 0
        
        for date, score in sentiment_scores.items():
            # Trades suspended if sentiment < threshold (paper's rule)
            allowed = score >= threshold
            sentiment_filter[date] = allowed
            
            if not allowed:
                blocked_days += 1
                print(f"   🚫 {date}: Sentiment {score:.3f} below threshold - BLOCKED")
        
        print(f"✅ Sentiment filter created: {blocked_days}/{len(sentiment_scores)} days blocked")
        
        return sentiment_filter
    
    def create_mock_news_data(self, days: int = 10) -> Dict[str, List[Dict]]:
        """
        Create mock news data for testing
        
        Args:
            days: Number of days to generate
        
        Returns:
            Mock news data
        """
        print(f"📝 Creating {days} days of mock news data...")
        
        # Sample news headlines
        positive_headlines = [
            "Solana network shows strong performance with new DeFi integrations",
            "SOL price surges as institutional interest grows",
            "Solana ecosystem expands with major partnership announcements",
            "Technical analysis suggests SOL bullish momentum continues",
            "Solana blockchain achieves new milestone in transaction speed"
        ]
        
        negative_headlines = [
            "Technical issues reported on Solana blockchain",
            "SOL experiences volatility amid market uncertainty",
            "Regulatory concerns impact Solana ecosystem",
            "Solana network congestion causes delays",
            "Major Solana project faces security challenges"
        ]
        
        neutral_headlines = [
            "Solana maintains current market position",
            "SOL trading volume remains stable",
            "Solana developers release routine network update",
            "Market analysts observe Solana price consolidation",
            "Solana ecosystem continues steady growth"
        ]
        
        news_data = {}
        base_date = datetime(2024, 1, 15)
        
        for i in range(days):
            current_date = base_date + pd.Timedelta(days=i)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Generate 2-5 articles per day
            num_articles = np.random.randint(2, 6)
            articles = []
            
            for j in range(num_articles):
                # Randomly choose sentiment
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                               p=[0.4, 0.3, 0.3])
                
                if sentiment_type == 'positive':
                    headline = np.random.choice(positive_headlines)
                elif sentiment_type == 'negative':
                    headline = np.random.choice(negative_headlines)
                else:
                    headline = np.random.choice(neutral_headlines)
                
                # Random time before 9:30 AM
                hour = np.random.randint(6, 10)
                minute = np.random.randint(0, 60)
                if hour == 9 and minute >= 30:
                    minute = np.random.randint(0, 30)
                
                articles.append({
                    'text': headline,
                    'time': time(hour, minute),
                    'source': 'Mock News',
                    'sentiment_type': sentiment_type
                })
            
            news_data[date_str] = articles
        
        print(f"✅ Mock news data created: {sum(len(v) for v in news_data.values())} total articles")
        
        return news_data
    
    def evaluate_sentiment_performance(self, sentiment_scores: Dict[str, float], 
                                    returns: Dict[str, float]) -> Dict:
        """
        Evaluate sentiment analysis performance against returns
        
        Args:
            sentiment_scores: Daily sentiment scores
            returns: Daily returns
        
        Returns:
            Performance metrics
        """
        # Align dates
        common_dates = set(sentiment_scores.keys()) & set(returns.keys())
        
        if not common_dates:
            return {'error': 'No common dates between sentiment and returns'}
        
        sentiment_aligned = [sentiment_scores[date] for date in sorted(common_dates)]
        returns_aligned = [returns[date] for date in sorted(common_dates)]
        
        # Calculate correlation
        correlation = np.corrcoef(sentiment_aligned, returns_aligned)[0, 1]
        
        # Calculate directional accuracy
        sentiment_direction = np.sign(sentiment_aligned)
        returns_direction = np.sign(returns_aligned)
        directional_accuracy = np.mean(sentiment_direction == returns_direction)
        
        # Calculate filter performance (paper's approach)
        threshold = -0.70
        filter_mask = np.array(sentiment_aligned) >= threshold
        filtered_returns = np.array(returns_aligned)[filter_mask]
        
        metrics = {
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'total_days': len(common_dates),
            'blocked_days': np.sum(np.array(sentiment_aligned) < threshold),
            'filtered_return_mean': np.mean(filtered_returns) if len(filtered_returns) > 0 else 0,
            'unfiltered_return_mean': np.mean(returns_aligned),
            'filter_improvement': np.mean(filtered_returns) - np.mean(returns_aligned) if len(filtered_returns) > 0 else 0
        }
        
        return metrics
    
    def save_sentiment_results(self, results: Dict, filepath: str):
        """Save sentiment analysis results"""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, bool):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_converted = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"✅ Results saved to {filepath}")


def main():
    """Test FinBERT sentiment analysis"""
    print("🚀 Testing FinBERT Sentiment Analysis...")
    
    # Initialize sentiment analyzer
    sentiment_analyzer = FinBERTSentiment()
    
    # Test single text analysis
    test_texts = [
        "Solana price surges to new heights as adoption increases",
        "Technical issues cause network congestion on Solana",
        "Solana maintains steady performance in volatile market"
    ]
    
    print("\n📝 Testing single text analysis:")
    for text in test_texts:
        result = sentiment_analyzer.analyze_text_sentiment(text)
        print(f"   Text: {text[:50]}...")
        print(f"   Sentiment: {result['label']} ({result['confidence']:.3f})")
        print(f"   Score: {result['normalized_score']:.3f}")
        print()
    
    # Create mock news data
    mock_news = sentiment_analyzer.create_mock_news_data(days=5)
    
    # Aggregate daily sentiment
    daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(mock_news)
    
    # Create sentiment filter
    sentiment_filter = sentiment_analyzer.create_sentiment_filter(daily_sentiment)
    
    # Test batch analysis
    all_articles = []
    for articles in mock_news.values():
        all_articles.extend([article['text'] for article in articles])
    
    batch_results = sentiment_analyzer.analyze_batch_sentiment(all_articles[:10])
    
    print(f"\n📊 Batch analysis results: {len(batch_results)} articles")
    
    # Save results
    results = {
        'daily_sentiment': daily_sentiment,
        'sentiment_filter': sentiment_filter,
        'batch_results': batch_results[:3]  # Save first 3 for demo
    }
    
    sentiment_analyzer.save_sentiment_results(results, 'logs/sentiment_test_results.json')
    
    print("\n🎉 FinBERT sentiment analysis test complete!")
    return sentiment_analyzer


if __name__ == "__main__":
    analyzer = main()
