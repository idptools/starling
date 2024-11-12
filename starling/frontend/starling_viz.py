import matplotlib.pyplot as plt

def plot_matrices(original, computed, difference, filename):
    """Plot the original, computed, and difference matrices using imshow and save to disk."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the original distance matrix
    im0 = axs[0].imshow(original, cmap="viridis")
    axs[0].set_title("Original Distance Matrix")
    axs[0].set_xlabel("Residue Index")
    axs[0].set_ylabel("Residue Index")
    fig.colorbar(im0, ax=axs[0])

    # Plot the computed distance matrix
    im1 = axs[1].imshow(computed, cmap="viridis")
    axs[1].set_title("Computed Distance Matrix (GD)")
    axs[1].set_xlabel("Residue Index")
    axs[1].set_ylabel("Residue Index")
    fig.colorbar(im1, ax=axs[1])

    # Plot the difference matrix
    im2 = axs[2].imshow(difference, cmap="viridis")
    axs[2].set_title("Difference Matrix (GD)")
    axs[2].set_xlabel("Residue Index")
    axs[2].set_ylabel("Residue Index")
    fig.colorbar(im2, ax=axs[2])

    # Save the plot to disk
    plt.savefig(filename)
    plt.close()
