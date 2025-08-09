import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import time
from tqdm import tqdm

# Import our modules
from src.handwritten_text_rnn import CharRNN, visualize_text
from src.generate_dataset import generate_handwritten_dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def char_tensor(string, all_chars):
    """Convert string to character tensor"""
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_chars.index(string[c])
        except ValueError:
            # Handle characters not in all_chars
            tensor[c] = 0  # Use first character as default
    return tensor

def random_training_set(chunk_len, batch_size, text, all_chars):
    """Generate random training batch"""
    inp = torch.zeros(batch_size, chunk_len).long()
    target = torch.zeros(batch_size, chunk_len).long()
    
    for bi in range(batch_size):
        start_index = random.randint(0, len(text) - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = text[start_index:end_index]
        
        inp[bi] = char_tensor(chunk[:-1], all_chars)
        target[bi] = char_tensor(chunk[1:], all_chars)
    
    return inp, target

def generate(model, prime_str='A', predict_len=100, temperature=0.8, all_chars=None):
    """Generate text from the model"""
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

def train(model, text, all_chars, n_epochs, batch_size=32, chunk_len=200, print_every=100, plot_every=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    all_losses = []
    loss_avg = 0
    
    for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
        inp, target = random_training_set(chunk_len, batch_size, text, all_chars)
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        # Zero gradients
        optimizer.zero_grad()
        loss = 0
        
        # Forward pass
        for c in range(chunk_len):
            output, hidden = model(inp[:, c].unsqueeze(1), hidden)
            loss += criterion(output.squeeze(1), target[:, c])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print loss
        loss_avg += loss.item() / chunk_len
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{n_epochs}, Loss: {loss_avg:.4f}')
            loss_avg = 0
        
        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0
    
    # Plot the training loss
    plt.figure()
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs (x{})'.format(plot_every))
    plt.ylabel('Loss')
    plt.savefig('models/loss_plot.png')
    
    return model, all_losses

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic handwritten text dataset...")
    text = generate_handwritten_dataset(num_samples=50)
    
    # If the dataset generation function returns None, load from file
    if not text:
        with open('data/handwritten_samples/handwritten_corpus.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Define character set
    all_chars = string.printable
    n_chars = len(all_chars)
    
    print(f"Data loaded. Corpus length: {len(text)}")
    print(f"Character set size: {n_chars}")
    
    # Model parameters
    hidden_size = 128
    n_layers = 2
    batch_size = 32
    chunk_len = 200
    n_epochs = 500  # Reduced for demonstration
    
    # Initialize model
    model = CharRNN(n_chars, hidden_size, n_chars, n_layers)
    print(f"Model initialized with {hidden_size} hidden units and {n_layers} layers")
    
    # Train model
    print("Starting training...")
    model, losses = train(model, text, all_chars, n_epochs, batch_size, chunk_len)
    
    # Save model
    model_path = 'models/handwritten_rnn.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate text
    print("Generating handwritten-like text...")
    seed_texts = ["The quick", "Hello", "Once upon", "In the"]
    
    for i, seed in enumerate(seed_texts):
        generated_text = generate(model, prime_str=seed, predict_len=200, temperature=0.8, all_chars=all_chars)
        print(f"\nGenerated text from seed '{seed}':")
        print(generated_text)
        
        # Visualize the generated text
        output_path = f'output/generated_text_{i+1}.png'
        visualize_text(generated_text, output_path)

if __name__ == "__main__":
    main()