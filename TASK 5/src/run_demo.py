import os
import torch
import matplotlib.pyplot as plt
import string
import sys

# Add the src directory to the path
sys.path.append('.')

# Import our modules
from src.handwritten_text_rnn import CharRNN, visualize_text

def load_model(model_path, hidden_size=128, n_layers=2):
    """Load a trained model"""
    # Define character set
    all_chars = string.printable
    n_chars = len(all_chars)
    
    # Initialize model
    model = CharRNN(n_chars, hidden_size, n_chars, n_layers)
    
    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    return model, all_chars

def generate_text(model, prime_str, predict_len=100, temperature=0.8, all_chars=None):
    """Generate text from the model"""
    model.eval()  # Set model to evaluation mode
    
    # Convert prime string to tensor
    def char_tensor(string, all_chars):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = all_chars.index(string[c])
            except ValueError:
                tensor[c] = 0  # Use first character as default
        return tensor
    
    hidden = model.init_hidden(1)
    prime_input = char_tensor(prime_str, all_chars).unsqueeze(0)
    
    predicted = prime_str
    
    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:, p].unsqueeze(1), hidden)
    
    inp = prime_input[:, -1].unsqueeze(1)
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_chars[top_i]
        predicted += predicted_char
        inp = torch.tensor([[top_i]], dtype=torch.long)
    
    return predicted

def main():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Load model
    model_path = 'models/handwritten_rnn.pth'
    
    # If model doesn't exist, train a simple one
    if not os.path.exists(model_path):
        print("No trained model found. Running training script first...")
        from src.train_and_generate import main as train_main
        train_main()
    
    # Load the model
    model, all_chars = load_model(model_path)
    
    # Generate text with different seeds
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
        generated_text = generate_text(model, seed, predict_len=200, temperature=0.7, all_chars=all_chars)
        print(f"Generated text: {generated_text[:50]}...")
        
        # Visualize the generated text
        output_path = f'output/demo_text_{i+1}.png'
        visualize_text(generated_text, output_path)
        print(f"Visualization saved to {output_path}")
    
    print("\nDemo completed! Check the 'output' directory for visualizations.")

if __name__ == "__main__":
    main()