import torch
from torch.utils.data import Dataset, DataLoader
from model import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, csv_file):
        data_frame = pd.read_csv(csv_file, header=None, skiprows=1, nrows=3200)

        self.samples = data_frame.iloc[:, :-1].values  # Normalize data to [0,1]
        self.labels = data_frame.iloc[:, -1].values

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]).float(), torch.tensor(self.labels[idx]).long()

batch_size = 32
data_path = "standardized_data.csv"
dataset = MyDataset(data_path)
dataloader = DataLoader(dataset, batch_size, shuffle=True)


input_dim = 100
output_dim = 1

generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(output_dim).to(device)

criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.00005, betas=(0, 0.9))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0, 0.9))

num_epochs = 150

checkpoint_dir = "./checkpoints2/"
checkpoint_list = "./checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Training loop
best_disc_loss = float('inf')
best_gen_loss = float('inf')
disc_loss_history = []
gen_loss_history = []

for epoch in range(num_epochs):
    for i, (real_data, labels) in enumerate(dataloader):
        real_data = real_data.unsqueeze(1).to(device)
        labels = labels.to(device)
        batch_size = real_data.shape[0]

        # Train the discriminator
        discriminator_optimizer.zero_grad()

        # Real data
        real_output = discriminator(real_data, labels)
        real_loss = -torch.mean(real_output)  # Note the negative sign

        # Fake data
        noise = torch.randn(batch_size, input_dim, 1).to(device)
        fake_data = generator(noise, labels)
        fake_output = discriminator(fake_data.detach(), labels)
        fake_loss = torch.mean(fake_output)  # Note the absence of negative sign

        # Gradient penalty
        alpha = torch.rand(real_data.size(0), 1, 1).to(device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        disc_interpolates = discriminator(interpolates, labels)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        lambda_gp = 10  # Gradient penalty coefficient
        gradient_penalty_loss = lambda_gp * gradient_penalty

        discriminator_loss = real_loss + fake_loss + gradient_penalty_loss

        # Backpropagation for discriminator
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Store loss history for plotting
        disc_loss_history.append(discriminator_loss.item())

        # Train the generator
        if i % 5 == 0:  # Update the generator every 5 iterations
            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, input_dim, 1).to(device)
            fake_data = generator(noise, labels)
            fake_output = discriminator(fake_data, labels)
            generator_loss = -torch.mean(fake_output)  # Note the negative sign
            generator_loss.backward()
            generator_optimizer.step()

            # Store loss history for plotting
            gen_loss_history.append(generator_loss.item())

        if i == len(dataloader) - 1:  # Check if it's the last iteration
            # Save best models
            if discriminator_loss.item() < best_disc_loss or generator_loss.item() < best_gen_loss:
                if discriminator_loss.item() < best_disc_loss:
                    best_disc_loss = discriminator_loss.item()
                    torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "best_discriminator.pth"))
                if generator_loss.item() < best_gen_loss:
                    best_gen_loss = generator_loss.item()
                    torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "best_generator.pth"))
            if epoch % 10 == 0:
                torch.save(generator.state_dict(), os.path.join(checkpoint_list, f"generator_epoch_{epoch}.pth"))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_list, f"discriminator_epoch_{epoch}.pth"))
            if epoch % 50 == 0:
                with torch.no_grad():
                    # Generate a batch of fake samples
                    test_noise = torch.randn(1, input_dim, 1).to(device)
                    test_label = torch.randint(0, 2, (1,)).to(device)  # Just an example label; you can change as needed
                    generated_sample = generator(test_noise, test_label).squeeze().cpu().numpy()
                    # Plot the generated sample
                    plt.figure(figsize=(10, 5))
                    plt.title(f"Generated Sample at Epoch {epoch}")
                    plt.plot(generated_sample)
                    plt.grid(True)
                    plt.show()
            print(f"Epoch [{epoch}/{num_epochs}]"
                    f"D_loss: {discriminator_loss.item()}, G_loss: {generator_loss.item()}")

# Plotting the training loss after the loop
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_loss_history, label="Generator")
plt.plot(disc_loss_history, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
