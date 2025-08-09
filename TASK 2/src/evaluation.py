"""
Model Evaluation Module for Credit Card Fraud Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import pickle
import os

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model with comprehensive metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC-AUC if probabilities are available
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate average precision
        avg_precision = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.results[model_name]
    
    def evaluate_all_models(self, models, X_test, y_test):
        """
        Evaluate all models and return results
        """
        if models is None:
            # Load models from disk
            models = self.load_models()
        
        for model_name, model in models.items():
            try:
                print(f"Evaluating {model_name}...")
                self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        return self.results
    
    def load_models(self):
        """
        Load all saved models from disk
        """
        models = {}
        model_files = [
            'logistic_regression_model.pkl',
            'decision_tree_model.pkl',
            'random_forest_model.pkl',
            'svm_model.pkl',
            'xgboost_model.pkl'
        ]
        
        for model_file in model_files:
            model_path = f'models/{model_file}'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('_model.pkl', '')
                    models[model_name] = pickle.load(f)
        
        return models
    
    def print_results(self):
        """
        Print evaluation results in a formatted table
        """
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'ROC-AUC': [],
            'Avg Precision': []
        })
        
        for model_name, metrics in self.results.items():
            results_df = results_df.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A",
                'Avg Precision': f"{metrics['avg_precision']:.4f}" if metrics['avg_precision'] else "N/A"
            }, ignore_index=True)
        
        print(results_df.to_string(index=False))
        print("="*80)
    
    def plot_confusion_matrices(self, save_path='models/confusion_matrices.png'):
        """
        Plot confusion matrices for all models
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def plot_roc_curves(self, save_path='models/roc_curves.png'):
        """
        Plot ROC curves for all models
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if metrics['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(metrics['probabilities'], metrics['predictions'])
                roc_auc = metrics['roc_auc']
                
                plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curves saved to {save_path}")
    
    def plot_precision_recall_curves(self, save_path='models/precision_recall_curves.png'):
        """
        Plot precision-recall curves for all models
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if metrics['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(metrics['probabilities'], metrics['predictions'])
                avg_precision = metrics['avg_precision']
                
                plt.plot(recall, precision, label=f'{model_name.replace("_", " ").title()} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Models')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Precision-recall curves saved to {save_path}")
    
    def plot_metrics_comparison(self, save_path='models/metrics_comparison.png'):
        """
        Plot bar chart comparing key metrics across models
        """
        if not self.results:
            print("No evaluation results available for plotting.")
            return
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            model_names = []
            metric_values = []
            
            for model_name, metrics in self.results.items():
                model_names.append(model_name.replace('_', ' ').title())
                metric_values.append(metrics[metric])
            
            axes[i].bar(model_names, metric_values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(metric_values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Metrics comparison saved to {save_path}")
    
    def generate_report(self, save_path='models/evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report
        """
        if not self.results:
            print("No evaluation results available for report generation.")
            return
        
        with open(save_path, 'w') as f:
            f.write("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"MODEL: {model_name.replace('_', ' ').upper()}\n")
                f.write("-"*40 + "\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
                if metrics['roc_auc']:
                    f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
                if metrics['avg_precision']:
                    f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n")
                f.write("\n")
                
                # Confusion matrix
                f.write("Confusion Matrix:\n")
                f.write(str(metrics['confusion_matrix']) + "\n\n")
                
                # Classification report
                f.write("Classification Report:\n")
                f.write(classification_report(metrics['predictions'], metrics['predictions'])) + "\n\n"
        
        print(f"Evaluation report saved to {save_path}")
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model based on a specific metric
        """
        if not self.results:
            return None, None
        
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric] 