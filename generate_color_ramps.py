import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_and_save_colormap_image(type_name, color_hex, save_path):
    """Creates a colormap from white to a given color and saves it as a PNG."""
    # Create a colormap that linearly interpolates from white to the specified color
    cmap = LinearSegmentedColormap.from_list(
        name=f'{type_name}_cmap',
        colors=['#FFFFFF', color_hex],
        N=256
    )
    
    # Create a gradient image to display the colormap
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 1)) # Adjust size as needed
    ax.imshow(gradient, aspect='auto', cmap=cmap)

    # Remove all axes, ticks, and labels for a clean schematic view
    ax.axis('off')

    # Ensure the layout is tight and has no padding
    plt.tight_layout(pad=0)

    # Save the figure
    output_filename = os.path.join(save_path, f'{type_name}_colormap.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved colormap for {type_name} to {output_filename}')

def main():
    """Main function to generate colormap images for all vehicle types."""
    # Define the vehicle types and their corresponding colors
    color_mapping = {
        'AV-AV': '#CFEAF1',
        'AV-HV': '#A9E9E4',
        'HV-HV': '#F3BAC3',
        'HV-AV': '#A5B1DA'
    }

    # Define the save path for the images
    save_path = 'fig/vis/mapping/color_ramps/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate and save a colormap image for each type
    for type_name, color_hex in color_mapping.items():
        create_and_save_colormap_image(type_name, color_hex, save_path)

if __name__ == '__main__':
    main()