import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from model3 import Generator

# Set random seed to ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the generator model
input_dim = 100  
output_dim = 1   
generator = Generator(input_dim, output_dim).to(device)
checkpoint_dir = "./checkpoints2/"
generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_generator.pth")))

# User input
label = int(input("Please enter the label (0 or 1): "))
while label not in [0, 1]:
    label = int(input("Invalid input. Please enter 0 or 1: "))

num_samples = int(input("Please enter the number of signals to generate (at least 32): "))
while num_samples < 32:
    num_samples = int(input("Invalid input. Please enter a number greater than or equal to 32: "))

# Generate signals
generator.eval()
with torch.no_grad():
    noise = torch.randn(num_samples, input_dim, 1).to(device)
    labels_tensor = torch.full((num_samples,), label, dtype=torch.long).to(device)
    generated_signals = generator(noise, labels_tensor).squeeze().cpu().numpy()

# Save to CSV
output_filename = f"generated_signals_label{label}_num{num_samples}.csv"
np.savetxt(output_filename, generated_signals, delimiter=",")

# Randomly select and plot 5 signals
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(5, 1, i+1)
    signal_index = random.randint(0, num_samples - 1)
    plt.plot(generated_signals[signal_index])
    plt.title(f"Generated Signal {signal_index+1}")
    plt.grid(True)
plt.tight_layout()
plt.show()

print(f"The generated signals have been saved to the file: {output_filename}")
