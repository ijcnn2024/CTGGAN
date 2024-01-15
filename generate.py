import torch
import numpy as np
import os
from model import Generator  # Ensure this import matches your model file and class

def generate_signals(generator, label, num_signals, input_dim):
    with torch.no_grad():
        noise = torch.randn(num_signals, input_dim, 1).to(device)
        labels_tensor = torch.full((num_signals,), label, dtype=torch.long).to(device)
        generated_signals = generator(noise, labels_tensor).squeeze().cpu().numpy()
    return generated_signals

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model parameters (adjust these as per your model's architecture)
    input_dim = 100
    output_dim = 1

    # Load the best saved model
    generator = Generator(input_dim, output_dim).to(device)
    checkpoint_dir = "./checkpoints2/"
    model_path = os.path.join(checkpoint_dir, "best_generator.pth")

    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")

    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # User input for label and number of signals
    label = int(input("Please enter the label (0 or 1): "))
    while label not in [0, 1]:
        label = int(input("Invalid input. Please enter 0 or 1: "))

    num_signals = int(input("Please enter the number of signals to generate: "))
    while num_signals <= 0:
        num_signals = int(input("Invalid input. Please enter a positive number: "))

    # Generate signals
    generated_signals = generate_signals(generator, label, num_signals, input_dim)

    # Save to CSV
    output_filename = f"generated_signals_label{label}_num{num_signals}.csv"
    np.savetxt(output_filename, generated_signals, delimiter=",")
    print(f"Generated signals saved to {output_filename}")
