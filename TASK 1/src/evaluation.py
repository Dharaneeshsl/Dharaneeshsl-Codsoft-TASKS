import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pickle
import os

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cm = confusion_matrix(y_test, y_pred)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test):
        if models is None:
            models = self.load_models()
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        
        return results
    
    def load_models(self):
        models = {}
        model_names = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
        
        for model_name in model_names:
            model_path = f'models/{model_name}_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        return models
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def plot_accuracy_comparison(self, save_path=None):
        if not self.results:
            print("No evaluation results available")
            return
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        if not self.results:
            print("No evaluation results available")
            return
        
        model_names = list(self.results.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        data = []
        for metric in metrics:
            values = [self.results[name][metric] for name in model_names]
            data.append(values)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.figure(figsize=(14, 8))
        
        for i, (metric, values) in enumerate(zip(metrics, data)):
            plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        plt.title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width, model_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def generate_report(self, save_path=None):
        if not self.results:
            print("No evaluation results available")
            return
        
        report = "MOVIE GENRE CLASSIFICATION - MODEL EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"
        report += f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
        report += "-" * 60 + "\n"
        
        for model_name, metrics in self.results.items():
            report += f"{model_name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
            report += f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n"
        
        report += "\n" + "=" * 60 + "\n\n"
        
        for model_name, metrics in self.results.items():
            report += f"DETAILED RESULTS FOR {model_name.upper()}:\n"
            report += "-" * 40 + "\n"
            report += f"Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"Precision: {metrics['precision']:.4f}\n"
            report += f"Recall: {metrics['recall']:.4f}\n"
            report += f"F1-Score: {metrics['f1_score']:.4f}\n"
            if metrics['roc_auc']:
                report += f"ROC AUC: {metrics['roc_auc']:.4f}\n"
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        print(report)
        return report
    
    def save_evaluation_results(self, filepath='evaluation_results.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath='evaluation_results.pkl'):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
            print(f"Evaluation results loaded from {filepath}")
        else:
            print(f"Evaluation results file not found: {filepath}")