# Handwritten Text Generation with RNN

This project implements a character-level recurrent neural network (RNN) to generate handwritten-like text. The model learns patterns from handwritten text examples and generates new text that mimics handwriting styles.

## Project Structure

```
.
├── data/
│   └── handwritten_samples/      # Generated handwritten text samples
│       └── handwritten_corpus.txt # Full text corpus
├── models/
│   ├── handwritten_rnn.pth       # Trained model weights
│   └── loss_plot.png             # Training loss visualization
├── output/
│   └── generated_text_*.png      # Visualizations of generated text
├── src/
│   ├── generate_dataset.py       # Script to generate synthetic handwritten text dataset
│   ├── handwritten_text_rnn.py   # Main RNN model implementation
│   ├── train_and_generate.py     # Script to train the model and generate text
│   └── run_demo.py               # Demo script to showcase the model
└── README.md                     # Project documentation
```

## Features

- Character-level RNN for text generation
- Synthetic handwritten text dataset generation
- Text visualization to mimic handwriting styles
- Customizable text generation parameters

## How It Works

1. **Data Generation**: The system creates a synthetic dataset of handwritten-like text examples.
2. **Model Architecture**: A character-level RNN learns patterns from the handwritten text.
3. **Training Process**: The model is trained to predict the next character in a sequence.
4. **Text Generation**: Given a seed text, the model generates new handwritten-like text.
5. **Visualization**: The generated text is visualized to mimic handwriting styles.

## Model Architecture

The character-level RNN consists of:
- An embedding layer to convert characters to vectors
- RNN layers to capture sequential patterns
- A linear decoder to predict the next character
- Temperature-based sampling for text generation

## Usage

### Running the Demo

To run the demo and generate handwritten-like text:

```bash
python src/run_demo.py
```

This will:
1. Load the trained model (or train one if not available)
2. Generate text from various seed phrases
3. Visualize the generated text to look like handwriting
4. Save the visualizations in the `output` directory

### Training a New Model

To train a new model from scratch:

```bash
python src/train_and_generate.py
```

This will:
1. Generate a synthetic dataset of handwritten text examples
2. Train the RNN model on this dataset
3. Save the trained model to `models/handwritten_rnn.pth`
4. Generate sample text using the trained model

## Customization

You can customize various aspects of the model:

- **Model Parameters**: Adjust hidden size, number of layers, etc. in `train_and_generate.py`
- **Training Parameters**: Modify batch size, epochs, learning rate in `train_and_generate.py`
- **Generation Parameters**: Change temperature, seed text, prediction length in `run_demo.py`
- **Visualization Style**: Modify the visualization parameters in `handwritten_text_rnn.py`

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Future Improvements

- Implement more advanced RNN architectures (LSTM, GRU)
- Use real handwritten text datasets for training
- Add style transfer capabilities for different handwriting styles
- Implement attention mechanisms for better text generation
- Create a web interface for interactive text generation