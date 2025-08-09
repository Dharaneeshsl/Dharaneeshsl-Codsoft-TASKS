"""
Simple demo of handwritten text generation concept
This script demonstrates the concept without requiring full model training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import string
from matplotlib.font_manager import FontProperties

def generate_handwritten_text(text, output_path='output/simple_demo.png'):
    """Generate a visualization of text that looks like handwriting"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # Use default font with slight variations in style
    font_styles = ['normal', 'italic']
    font_weights = ['normal', 'bold']
    
    # Create font with random style and weight
    font = FontProperties(style=random.choice(font_styles), 
                         weight=random.choice(font_weights))
    
    # Split text into lines
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) > 60:  # Limit line length
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Plot each line with slight variations to simulate handwriting
    y_position = 0.9
    for line in lines:
        # Add slight random variations to simulate handwriting
        x_jitter = np.random.uniform(-0.005, 0.005, len(line))
        y_jitter = np.random.uniform(-0.005, 0.005, len(line))
        
        for i, char in enumerate(line):
            plt.text(0.1 + i*0.01 + x_jitter[i], y_position + y_jitter[i], 
                     char, fontsize=14, fontproperties=font)
        
        y_position -= 0.05
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def simple_character_model(seed_text, length=100):
    """
    A very simple character model that demonstrates the concept
    without requiring training a neural network
    """
    # Define character transition probabilities (simplified)
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    space_chars = ' '
    punct_chars = ',.!?;:'
    
    # Simple rules for character generation
    text = seed_text
    prev_char = seed_text[-1].lower()
    
    for _ in range(length):
        if prev_char in vowels:
            # After a vowel, likely a consonant or space
            choices = consonants + space_chars
            weights = [3] * len(consonants) + [1] * len(space_chars)
        elif prev_char in consonants:
            # After a consonant, likely a vowel or space
            choices = vowels + space_chars
            weights = [4] * len(vowels) + [1] * len(space_chars)
        elif prev_char in space_chars:
            # After a space, likely a consonant or vowel to start a new word
            choices = consonants + vowels
            weights = [3] * len(consonants) + [2] * len(vowels)
        elif prev_char in punct_chars:
            # After punctuation, likely a space
            choices = space_chars
            weights = [1] * len(space_chars)
        else:
            # Default case
            choices = vowels + consonants + space_chars
            weights = [1] * (len(vowels) + len(consonants) + len(space_chars))
        
        # Add occasional punctuation
        if random.random() < 0.02 and prev_char not in punct_chars and prev_char not in space_chars:
            choices += punct_chars
            weights += [0.5] * len(punct_chars)
        
        # Select next character
        next_char = random.choices(choices, weights=weights)[0]
        
        # Capitalize after period and space
        if len(text) >= 2 and text[-2:] == '. ':
            next_char = next_char.upper()
        
        text += next_char
        prev_char = next_char
    
    return text

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Seed texts
    seed_texts = [
        "The quick brown fox",
        "Hello world",
        "Once upon a time",
        "In the beginning",
        "Dear friend"
    ]
    
    # Generate and visualize text for each seed
    for i, seed in enumerate(seed_texts):
        print(f"\nGenerating text from seed: '{seed}'")
        
        # Generate text using simple model
        generated_text = simple_character_model(seed, length=150)
        print(f"Generated text: {generated_text}")
        
        # Visualize the generated text
        output_path = f'output/simple_demo_{i+1}.png'
        visualize_path = generate_handwritten_text(generated_text, output_path)
    
    print("\nDemo completed! Check the 'output' directory for visualizations.")

if __name__ == "__main__":
    main()