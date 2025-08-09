import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description='Handwritten Text Generation with RNN')
    parser.add_argument('--action', type=str, default='simple-demo', 
                        choices=['train', 'generate', 'evaluate', 'demo', 'simple-demo'],
                        help='Action to perform: train, generate, evaluate, demo, or simple-demo')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (for train action)')
    parser.add_argument('--seed', type=str, default='The quick',
                        help='Seed text for generation (for generate action)')
    parser.add_argument('--length', type=int, default=200,
                        help='Length of text to generate (for generate action)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (for generate action)')
    args = parser.parse_args()
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    if args.action == 'train':
        print("Training the handwritten text RNN model...")
        from src.train_and_generate import main as train_main
        train_main()
        
    elif args.action == 'generate':
        print(f"Generating handwritten text with seed: '{args.seed}'")
        from src.handwritten_text_rnn import visualize_text
        from src.run_demo import load_model, generate_text
        
        model_path = 'models/handwritten_rnn.pth'
        if not os.path.exists(model_path):
            print("No trained model found. Training a new model first...")
            from src.train_and_generate import main as train_main
            train_main()
        
        model, all_chars = load_model(model_path)
        generated_text = generate_text(model, args.seed, predict_len=args.length, 
                                      temperature=args.temperature, all_chars=all_chars)
        
        print("\nGenerated text:")
        print(generated_text)
        
        output_path = 'output/generated_text.png'
        visualize_text(generated_text, output_path)
        print(f"\nVisualization saved to {output_path}")
        
    elif args.action == 'evaluate':
        print("Evaluating the handwritten text RNN model...")
        from src.evaluate_model import main as evaluate_main
        evaluate_main()
        
    elif args.action == 'demo':
        print("Running handwritten text generation demo...")
        from src.run_demo import main as demo_main
        demo_main()
    
    elif args.action == 'simple-demo':
        print("Running simple handwritten text generation demo...")
        from src.simple_demo import main as simple_demo_main
        simple_demo_main()
    
    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()