import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(loss_file, output_dir):
    """
    Plots the training loss and saves the plot.

    Args:
        loss_file (str): Path to the train_losses.npy file.
        output_dir (str): Directory to save the plot image.
    """
    if not os.path.exists(loss_file):
        print(f"Error: Loss file not found at {loss_file}")
        return

    # Load the loss values
    losses = np.load(loss_file)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_dir, 'training_loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss.')
    parser.add_argument('--loss_file', type=str, required=True,
                        help='Path to the train_losses.npy file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the plot image.')

    args = parser.parse_args()

    plot_loss(args.loss_file, args.output_dir)
