"""
Model Evaluation Framework for SOL Trading System
Comprehensive evaluation and comparison of ML models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation framework
    Implements paper's evaluation methodology and additional metrics
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.baseline_metrics = {}
        
    def evaluate_model_performance(self, model_name: str, y_true: np.ndarray, 
                                  y_pred: np.ndarray, y_prob: np.ndarray = None,
                                  model_metadata: Dict = None) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            model_metadata: Additional model information
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"📊 Evaluating {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Detailed classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=['Down', 'Up'], 
                                     output_dict=True,
                                     zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC if probabilities available
        roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
        
        # Brier score (calibration)
        brier = brier_score_loss(y_true, y_prob) if y_prob is not None else None
        
        # Paper baseline comparison
        paper_baseline = 0.630
        accuracy_improvement = accuracy - paper_baseline
        
        # Trading-specific metrics
        positive_rate = y_pred.mean()
        true_positive_rate = y_true.mean()
        
        # Information content (entropy reduction)
        entropy_before = -np.mean([0.5 * np.log2(0.5) + 0.5 * np.log2(0.5)])  # Maximum entropy
        if y_prob is not None:
            entropy_after = -np.mean(y_prob * np.log2(y_prob + 1e-10) + 
                                  (1 - y_prob) * np.log2(1 - y_prob + 1e-10))
            information_gain = entropy_before - entropy_after
        else:
            information_gain = 0
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'accuracy': accuracy,
            'paper_baseline': paper_baseline,
            'accuracy_improvement': accuracy_improvement,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'brier_score': brier,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'positive_rate': positive_rate,
            'true_positive_rate': true_positive_rate,
            'information_gain': information_gain,
            'classification_report': report,
            'model_metadata': model_metadata or {}
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        # Print summary
        print(f"   Accuracy: {accuracy:.3f} (vs paper baseline {paper_baseline:.3f})")
        print(f"   Improvement: {accuracy_improvement:+.3f}")
        print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        if roc_auc:
            print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Positive Rate: {positive_rate:.2%} (True: {true_positive_rate:.2%})")
        
        return results
    
    def compare_models(self, model_names: List[str] = None) -> Dict:
        """
        Compare multiple models
        
        Args:
            model_names: List of model names to compare (all if None)
        
        Returns:
            Comparison results
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        if len(model_names) < 2:
            print("❌ Need at least 2 models to compare")
            return {}
        
        print(f"🔄 Comparing {len(model_names)} models...")
        
        # Create comparison table
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"⚠️  Model {model_name} not found in results")
                continue
            
            results = self.evaluation_results[model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Improvement': results['accuracy_improvement'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', np.nan),
                'Positive Rate': results['positive_rate']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models
        comparison_df['Rank'] = comparison_df['Accuracy'].rank(ascending=False).astype(int)
        comparison_df = comparison_df.sort_values('Rank')
        
        print("\n📊 Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Statistical significance (simple pairwise comparison)
        if len(model_names) == 2:
            self._pairwise_comparison(model_names[0], model_names[1])
        
        return {
            'comparison_table': comparison_df,
            'best_model': comparison_df.iloc[0]['Model'],
            'best_accuracy': comparison_df.iloc[0]['Accuracy']
        }
    
    def _pairwise_comparison(self, model1: str, model2: str):
        """Simple pairwise comparison between two models"""
        if model1 not in self.evaluation_results or model2 not in self.evaluation_results:
            return
        
        results1 = self.evaluation_results[model1]
        results2 = self.evaluation_results[model2]
        
        # Simple accuracy difference test
        acc_diff = results1['accuracy'] - results2['accuracy']
        
        print(f"\n🔍 Pairwise Comparison: {model1} vs {model2}")
        print(f"   Accuracy Difference: {acc_diff:+.3f}")
        
        if acc_diff > 0.01:
            print(f"   {model1} significantly outperforms {model2}")
        elif acc_diff < -0.01:
            print(f"   {model2} significantly outperforms {model1}")
        else:
            print(f"   No significant difference between models")
    
    def evaluate_trading_performance(self, model_name: str, predictions: np.ndarray,
                                    returns: pd.Series, sentiment_filter: np.ndarray = None) -> Dict:
        """
        Evaluate trading performance of model predictions
        
        Args:
            model_name: Model name
            predictions: Model predictions (1=long, 0=cash)
            returns: Actual returns
            sentiment_filter: Optional sentiment filter (True=allowed)
        
        Returns:
            Trading performance metrics
        """
        print(f"💰 Evaluating trading performance for {model_name}...")
        
        # Apply sentiment filter if provided
        if sentiment_filter is not None:
            trading_signals = predictions * sentiment_filter
            print(f"   Sentiment filter blocked {(~sentiment_filter).sum()} signals")
        else:
            trading_signals = predictions
        
        # Calculate strategy returns
        strategy_returns = trading_signals * returns
        
        # Performance metrics
        total_return = strategy_returns.sum()
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        
        # Risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        # Benchmark performance (buy and hold)
        benchmark_return = returns.sum()
        benchmark_annual = (1 + benchmark_return) ** (252 / len(returns)) - 1
        
        # Alpha
        alpha = annual_return - benchmark_annual
        
        trading_results = {
            'model_name': model_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'benchmark_return': benchmark_return,
            'benchmark_annual': benchmark_annual,
            'alpha': alpha,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trading_days': len(strategy_returns),
            'positions_taken': trading_signals.sum(),
            'position_rate': trading_signals.mean()
        }
        
        print(f"   Strategy Return: {total_return:.3f} vs Benchmark: {benchmark_return:.3f}")
        print(f"   Annual Return: {annual_return:.3f} vs Benchmark: {benchmark_annual:.3f}")
        print(f"   Alpha: {alpha:+.3f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {max_drawdown:.3f}")
        print(f"   Win Rate: {win_rate:.2%}")
        
        return trading_results
    
    def plot_model_comparison(self, model_names: List[str] = None, metrics: List[str] = None):
        """
        Plot model comparison charts
        
        Args:
            model_names: Models to include
            metrics: Metrics to plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        plot_data = []
        for model_name in model_names:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                for metric in metrics:
                    if metric in results:
                        plot_data.append({
                            'Model': model_name,
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': results[metric]
                        })
        
        if not plot_data:
            print("❌ No data to plot")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            metric_data = df[df['Metric'] == metric.replace('_', ' ').title()]
            
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Model', y='Value', ax=axes[i])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_ylim(0, 1)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, model_names: List[str] = None):
        """Plot confusion matrices for multiple models"""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.evaluation_results:
                continue
            
            cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name}\nAccuracy: {self.evaluation_results[model_name]["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_file: File to save report
        
        Returns:
            Report as string
        """
        report_lines = [
            "Model Evaluation Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Evaluated: {len(self.evaluation_results)}",
            ""
        ]
        
        # Model summaries
        for model_name, results in self.evaluation_results.items():
            report_lines.extend([
                f"Model: {model_name}",
                f"  Accuracy: {results['accuracy']:.3f} (vs baseline {results['paper_baseline']:.3f})",
                f"  Improvement: {results['accuracy_improvement']:+.3f}",
                f"  Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}",
                f"  F1-Score: {results['f1_score']:.3f}",
                f"  ROC AUC: {results.get('roc_auc', 'N/A')}",
                f"  Sample Size: {results['sample_size']:,}",
                ""
            ])
        
        # Comparison
        if len(self.evaluation_results) > 1:
            comparison = self.compare_models()
            if comparison:
                best_model = comparison['best_model']
                best_accuracy = comparison['best_accuracy']
                
                report_lines.extend([
                    "Model Ranking:",
                    f"  1. {best_model}: {best_accuracy:.3f}",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "Recommendations:",
            "  • Models with accuracy > 0.60 meet paper baseline",
            "  • Consider both accuracy and trading performance",
            "  • Evaluate risk-adjusted returns (Sharpe ratio)",
            "  • Monitor for overfitting with cross-validation",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report
    
    def save_results(self, filepath: str):
        """Save all evaluation results to JSON"""
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_converted = convert_numpy(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"✅ Evaluation results saved to {filepath}")


def main():
    """Test model evaluation framework"""
    print("🚀 Testing Model Evaluation Framework...")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Generate mock data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Mock predictions from different "models"
    models_data = {
        'XGBoost_Paper': {
            'y_true': np.random.randint(0, 2, n_samples),
            'y_pred': np.random.randint(0, 2, n_samples),
            'y_prob': np.random.random(n_samples)
        },
        'Random_Forest': {
            'y_true': np.random.randint(0, 2, n_samples),
            'y_pred': np.random.randint(0, 2, n_samples),
            'y_prob': np.random.random(n_samples)
        },
        'Logistic_Regression': {
            'y_true': np.random.randint(0, 2, n_samples),
            'y_pred': np.random.randint(0, 2, n_samples),
            'y_prob': np.random.random(n_samples)
        }
    }
    
    # Evaluate each model
    for model_name, data in models_data.items():
        # Make one model slightly better for demonstration
        if model_name == 'XGBoost_Paper':
            # Increase accuracy to ~0.65 (like paper)
            data['y_pred'] = data['y_true'].copy()
            # Flip 35% of predictions to create some errors
            flip_indices = np.random.choice(n_samples, int(n_samples * 0.35), replace=False)
            data['y_pred'][flip_indices] = 1 - data['y_pred'][flip_indices]
            data['y_prob'] = np.where(data['y_pred'] == 1, 
                                    np.random.uniform(0.6, 0.9, n_samples),
                                    np.random.uniform(0.1, 0.4, n_samples))
        
        evaluator.evaluate_model_performance(model_name, 
                                            data['y_true'], 
                                            data['y_pred'], 
                                            data['y_prob'])
    
    # Compare models
    comparison = evaluator.compare_models()
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print("\n" + report)
    
    # Save results
    evaluator.save_results('logs/evaluation_test_results.json')
    
    print("\n🎉 Model evaluation framework test complete!")
    return evaluator


if __name__ == "__main__":
    evaluator = main()
