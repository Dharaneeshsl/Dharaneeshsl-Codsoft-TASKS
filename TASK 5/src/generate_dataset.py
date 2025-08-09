import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import random
import string

def generate_handwritten_dataset(num_samples=100, output_dir='data/handwritten_samples'):
    """
    Generate a synthetic dataset of handwritten-like text samples
    
    Args:
        num_samples: Number of text samples to generate
        output_dir: Directory to save the generated samples
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample sentences to use as base content
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a sample of handwritten text.",
        "Practice makes perfect when learning to write neatly.",
        "My handwriting varies depending on how quickly I write.",
        "Sometimes letters slant to the right when I'm in a hurry.",
        "Careful penmanship requires patience and attention to detail.",
        "The lazy dog slept under the shade of an old oak tree.",
        "Writing by hand is becoming a lost art in the digital age.",
        "Each person's handwriting is unique like a fingerprint.",
        "I need to improve my cursive writing skills with practice.",
        "The rain in Spain falls mainly on the plain.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "A journey of a thousand miles begins with a single step.",
        "Actions speak louder than words."
    ]
    
    # Available handwriting-like fonts
    handwriting_fonts = [
        'cursive',
        'monospace',  # Not handwriting but useful for variety
        'fantasy'     # Not handwriting but useful for variety
    ]
    
    # Generate text file with all samples
    all_text = ""
    
    # Generate samples
    for i in range(num_samples):
        # Select random sentences to combine
        num_sentences = random.randint(1, 3)
        selected_sentences = random.sample(base_sentences, num_sentences)
        text = " ".join(selected_sentences)
        
        # Add some random variations
        if random.random() < 0.3:
            # Add random punctuation errors
            if "," in text:
                text = text.replace(",", "", 1)
            elif "." in text:
                text = text.replace(".", "!", 1)
        
        # Add to the full text corpus
        all_text += text + " "
        
        # Create a visualization of the text
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        
        # Choose a random font
        font = FontProperties(family=random.choice(handwriting_fonts))
        
        # Add random variations to simulate handwriting
        x_jitter = np.random.uniform(-0.01, 0.01, len(text))
        y_jitter = np.random.uniform(-0.01, 0.01, len(text))
        
        # Split text into lines if it's too long
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) > 50:  # Limit line length
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Plot each line with variations
        y_position = 0.8
        for line in lines:
            for j, char in enumerate(line):
                plt.text(0.1 + j*0.015 + x_jitter[j % len(x_jitter)], 
                         y_position + y_jitter[j % len(y_jitter)], 
                         char, fontsize=14, fontproperties=font)
            y_position -= 0.2
        
        # Save the image
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{i+1}.png', dpi=150)
        plt.close()
    
    # Save the full text corpus
    with open(f'{output_dir}/handwritten_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(all_text)
    
    print(f"Generated {num_samples} handwritten text samples in {output_dir}")
    print(f"Total corpus length: {len(all_text)} characters")
    
    return all_text

if __name__ == "__main__":
    # Generate a dataset with 50 samples
    generate_handwritten_dataset(num_samples=50)