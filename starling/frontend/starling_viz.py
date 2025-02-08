import matplotlib.pyplot as plt

AA_COLOR = {'Y': '#ff9d00',
            'W': '#ff9d00',
            'F': '#ff9d00',
            'A': '#171616',
            'L': '#171616',
            'M': '#171616',
            'I': '#171616',
            'V': '#171616',
            'Q': '#04700d',
            'N': '#04700d',
            'S': '#04700d',
            'T': '#04700d',
            'H': '#04700d',
            'G': '#04700d',
            'E': '#ff0d0d',
            'D': '#ff0d0d',
            'R': '#2900f5',
            'K': '#2900f5',
            'C': '#ffe70d',
            'P': '#cf30b7'}

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

import matplotlib.pyplot as plt
import numpy as np
from sparrow.data.amino_acids import VALID_AMINO_ACIDS


def plot_protein_arcs(sequence, 
                      links, 
                      color_dict=AA_COLOR, 
                      contacts=False,
                      norm_lower_thresh = 1,
                      number_freq = 10,
                      filename=None):
    
    """
    Plot a protein sequence with arcs connecting residues.

    Parameters
    ----------
    sequence : str
        The protein sequence to plot.

    links : list
        A list of links between residues. Each link is represented as a list
        with three elements: the start residue index, the end residue index,
        and the contact score. The contact score can either be a contact score 
        (i.e. value between 0 and 1) reporting on the fraction of the ensemble
        in which the residues are in contact. It can also be a normalized distance; 
        typically the ratio of the actual distances normalized by some limiting
        theoretical model distance.

    contacts : bool, optional
        If True, the contact scores are interpreted as contact scores. If False,
        the contact scores are interpreted as distances. Default is False.

    norm_lower_thresh : float, optional
        The threshold for the lower bound of the normalized distance. Default is 1.

    number_freq : int
        The frequency of the number labels on the x-axis. Default is 10.

    Returns
    -------
    None, but if filename is provided, the plot is saved to disk.

    """
    
    # initialize the plot
    fig, ax = plt.subplots(figsize=(len(sequence) / 4, 4))

    # Set up the plot
    ax.set_xlim(-1, len(sequence))
    ax.set_ylim(-1.75, 1.75)  # Adjusted limits for better visibility

    # this code sets every 10th residue to be labeled
    ax.set_xticks(range(len(sequence)))
    xtick_numberlabels = []
    for idx in range(1, len(sequence)+1):
        if idx % number_freq == 0:
            xtick_numberlabels.append(idx)
        else:
            xtick_numberlabels.append('')            
    ax.set_xticklabels(xtick_numberlabels)  # Hide default labels
        
    # get unique links
    links = list(map(list, set(map(tuple, links))))

    # Color and write each residue to the plot
    for i, res in enumerate(sequence):
        ax.text(i, 0, res, fontsize=12, ha='center', va='center',
                color=color_dict.get(res, 'black'), fontweight='bold')
    

    # if there's at least one link to plot...
    if len(links) > 0:

        # if we passed contact socres (summed or averages)
        if contacts:
    
            # get all the link contact scores
            values = np.array(links)[:,2]
    
            # extract min and max
            min_val = min(values)
            max_val = max(values)
        
            # Avoid division by zero in case all values are the same 
            if max_val == min_val:
                return [0.7] * len(values)  # If all values are the same, map them to the midpoint
        
            # Normalize values to [0, 1], then invert and scale to [0.5, 0.9]
            recoded_links_vals = [0.5 + 0.4 * (max_val - v) / (max_val - min_val) for v in values]
    
            # finally, re-assign in the links list
            for i in range(len(links)):
                links[i][2] = recoded_links_vals[i]
            
        
        # Draw arcs
        idx = -1
        for start, end, score in links:
            idx = idx + 1
            if start >= len(sequence) or end >= len(sequence) or start < 0 or end < 0:
                continue  # Ensure valid indices
            
            mid = (start + end) / 2
            height = min(abs(end - start) / 15, 1)  # Reduce arc height further
    
    
            # if contacts all attractive and we alternative sides
            if contacts:
                color = '#310451'  # Attractive interaction
    
                if idx % 2 == 0:
                    y_offset = -0.1  # Bring arc closer to letters
                    arc = np.linspace(np.pi, 2 * np.pi, 100)  # Flip arc to face downward
                else:
                    y_offset = 0.1  # Bring arc closer to letters
                    arc = np.linspace(0, np.pi, 100)  # Standard arc shape
                    
                intensity = 1 - (score - 0.5) / 0.49  # Scale alpha (0.5 -> 1, 0.99 -> 0.1)
                width = 3 - 2.5 * (score - 0.5) / 0.49  # Scale width (0.5 -> 3, 0.99 -> 0.5)
    
    
            else:
                if score < norm_lower_thresh:
                    color = '#310451'  # Attractive interaction
                    y_offset = 0.1  # Bring arc closer to letters
                    arc = np.linspace(0, np.pi, 100)  # Standard arc shape
                    intensity = 1 - (score - 0.7) / 0.49  # Scale alpha (0.7 -> 1, 0.99 -> 0.1)
                    width = 3 - 2.5 * (score - 0.7) / 0.49  # Scale width (0.7 -> 3, 0.99 -> 0.5)
                else:
                    color = '#813C08'  # Repulsive interaction
                    y_offset = -0.1  # Bring arc closer to letters
                    arc = np.linspace(np.pi, 2 * np.pi, 100)  # Flip arc to face downward
                    intensity = 1 - ((1.5 - score) / 0.49)  # Scale alpha (1.5 -> 1, 1.01 -> 0.1)
                    width = 3 - 2.5 * ((1.5 - score) / 0.49)  # Scale width (1.5 -> 3, 1.01 -> 0.5)
                
            intensity = max(0.1, min(1, intensity))  # Clamp alpha values
            width = max(0.5, min(3, width))  # Clamp width values
            
            # Create arc
            x_arc = np.linspace(start, end, 100)
            y_arc = np.sin(arc) * height + y_offset

            # Create arc
            x_arc = np.linspace(start, end, 100)    
            y_arc = np.sin(arc) * height + y_offset
            ax.plot(x_arc, y_arc, color=color, alpha=intensity, linewidth=width)
        
    
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()  

