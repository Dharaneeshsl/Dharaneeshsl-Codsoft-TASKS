import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import string
import sys
from collections import Counter

sys.path.append('.')

from src.handwritten_text_rnn import CharRNN, visualize_text
from src.run_demo import load_model, generate_text

def calculate_character_diversity(text):
    if not text:
        return 0
    
    char_counts = Counter(text)
    total_chars = len(text)
    
    entropy = 0
    for char, count in char_counts.items():
        prob = count / total_chars
        entropy -= prob * np.log2(prob)
    
    max_entropy = np.log2(min(len(char_counts), total_chars))
    if max_entropy == 0:
        return 0
    
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def calculate_n_gram_diversity(text, n=2):
    if len(text) < n:
        return 0
    
    n_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
    n_gram_counts = Counter(n_grams)
    total_n_grams = len(n_grams)
    
    entropy = 0
    for n_gram, count in n_gram_counts.items():
        prob = count / total_n_grams
        entropy -= prob * np.log2(prob)
    
    max_entropy = np.log2(min(len(n_gram_counts), total_n_grams))
    if max_entropy == 0:
        return 0
    
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def evaluate_text_quality(generated_texts):
    results = {
        'char_diversity': [],
        'bigram_diversity': [],
        'trigram_diversity': [],
        'avg_length': [],
        'repetition_score': []
    }
    
    for text in generated_texts:
        char_div = calculate_character_diversity(text)
        results['char_diversity'].append(char_div)
        
        bigram_div = calculate_n_gram_diversity(text, n=2)
        trigram_div = calculate_n_gram_diversity(text, n=3)
        results['bigram_diversity'].append(bigram_div)
        results['trigram_diversity'].append(trigram_div)
        
        results['avg_length'].append(len(text))
        
        if len(text) >= 4:
            four_grams = [text[i:i+4] for i in range(len(text) - 4 + 1)]
            four_gram_counts = Counter(four_grams)
            repetition_score = sum(count - 1 for count in four_gram_counts.values()) / len(four_grams)
            results['repetition_score'].append(repetition_score)
        else:
            results['repetition_score'].append(0)
    
    avg_results = {
        'avg_char_diversity': np.mean(results['char_diversity']),
        'avg_bigram_diversity': np.mean(results['bigram_diversity']),
        'avg_trigram_diversity': np.mean(results['trigram_diversity']),
        'avg_text_length': np.mean(results['avg_length']),
        'avg_repetition_score': np.mean(results['repetition_score'])
    }
    
    return avg_results

def evaluate_model_at_temperatures(model, all_chars, seed_texts, temperatures=[0.5, 0.7, 1.0, 1.2]):
    results = {}
    
    for temp in temperatures:
        print(f"\nEvaluating at temperature = {temp}")
        generated_texts = []
        
        for seed in seed_texts:
            generated_text = generate_text(model, seed, predict_len=200, temperature=temp, all_chars=all_chars)
            generated_texts.append(generated_text)
            print(f"  Seed: '{seed}'\n  Generated: '{generated_text[:50]}...'\n")
        
        eval_results = evaluate_text_quality(generated_texts)
        results[temp] = eval_results
        
        print(f"  Character diversity: {eval_results['avg_char_diversity']:.4f}")
        print(f"  Bigram diversity: {eval_results['avg_bigram_diversity']:.4f}")
        print(f"  Trigram diversity: {eval_results['avg_trigram_diversity']:.4f}")
        print(f"  Repetition score: {eval_results['avg_repetition_score']:.4f}")
    
    return results

def plot_evaluation_results(results, output_path='output/evaluation_results.png'):
    temperatures = list(results.keys())
    
    metrics = [
        ('avg_char_diversity', 'Character Diversity'),
        ('avg_bigram_diversity', 'Bigram Diversity'),
        ('avg_trigram_diversity', 'Trigram Diversity'),
        ('avg_repetition_score', 'Repetition Score (lower is better)')
    ]
    
    plt.figure(figsize=(12, 10))
    
    for i, (metric, title) in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        values = [results[temp][metric] for temp in temperatures]
        plt.plot(temperatures, values, 'o-', linewidth=2)
        plt.title(title)
        plt.xlabel('Temperature')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Evaluation results plot saved to {output_path}")
    
    return output_path

def main():
    os.makedirs('output', exist_ok=True)
    
    model_path = 'models/handwritten_rnn.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found. Running training script first...")
        from src.train_and_generate import main as train_main
        train_main()
    
    model, all_chars = load_model(model_path)
    
    seed_texts = [
        "The quick",
        "Hello world",
        "Once upon a time",
        "In the beginning",
        "Dear friend"
    ]
    
    temperatures = [0.5, 0.7, 1.0, 1.2]
    results = evaluate_model_at_temperatures(model, all_chars, seed_texts, temperatures)
    
    plot_path = plot_evaluation_results(results)
    
    best_temp = max(temperatures, key=lambda t: 
                   results[t]['avg_char_diversity'] + 
                   results[t]['avg_bigram_diversity'] - 
                   results[t]['avg_repetition_score'])
    
    print(f"\nBest temperature found: {best_temp}")
    print("Generating final examples with best temperature...")
    
    for i, seed in enumerate(seed_texts):
        generated_text = generate_text(model, seed, predict_len=200, temperature=best_temp, all_chars=all_chars)
        output_path = f'output/best_text_{i+1}.png'
        visualize_text(generated_text, output_path)
    
    print("\nEvaluation completed! Check the 'output' directory for results.")

if __name__ == "__main__":
    main()