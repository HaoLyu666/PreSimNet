import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

def visualize_trajectory(trajectory, key, save_dir):
    """
    Visualize the position, velocity and gap of a single trajectory
    
    Args:
        trajectory: Trajectory data
        key: Trajectory key name
        save_dir: Save directory
    """
    # Get raw data
    raw_data = trajectory['raw_data']
    
    # Get trajectory type
    traj_type = trajectory['type']
    type_names = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}
    type_name = type_names[traj_type]
    
    # Create time steps array
    time_steps = np.arange(len(raw_data)) * 0.1  # Assume each time step is 0.1 second
    
    # Set font
    font_prop = FontProperties(family='Times New Roman')
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), dpi=300)
    
    # 1. Plot position
    axs[0].plot(time_steps, raw_data[:, 12], 'r-', linewidth=1.5, label='Leading Vehicle')
    axs[0].plot(time_steps, raw_data[:, 11], 'b-', linewidth=1.5, label='Following Vehicle')
    axs[0].set_xlabel('Time (s)', fontproperties=font_prop, fontsize=12)
    axs[0].set_ylabel('Position (m)', fontproperties=font_prop, fontsize=12)
    axs[0].legend(prop=font_prop, loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot velocity
    axs[1].plot(time_steps, raw_data[:, 4], 'r-', linewidth=1.5, label='Leading Vehicle')
    axs[1].plot(time_steps, raw_data[:, 6], 'b-', linewidth=1.5, label='Following Vehicle')
    axs[1].set_xlabel('Time (s)', fontproperties=font_prop, fontsize=12)
    axs[1].set_ylabel('Velocity (m/s)', fontproperties=font_prop, fontsize=12)
    axs[1].legend(prop=font_prop, loc='upper left')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot gap
    axs[2].plot(time_steps, raw_data[:, 8], 'g-', linewidth=1.5, label='Gap')
    axs[2].set_xlabel('Time (s)', fontproperties=font_prop, fontsize=12)
    axs[2].set_ylabel('Gap (m)', fontproperties=font_prop, fontsize=12)
    axs[2].legend(prop=font_prop, loc='upper left')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    plt.suptitle(f'Trajectory {key} ({type_name})', fontproperties=font_prop, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save image
    plt.savefig(os.path.join(save_dir, f'{key}.png'), bbox_inches='tight')
    plt.close()

def main():
    # Load trajectory data
    print("Loading trajectory data...")
    trajectory_file = "trajectory_samples/selected_trajectories.npy"
    if not os.path.exists(trajectory_file):
        print(f"Error: File not found {trajectory_file}")
        return
    
    trajectories = np.load(trajectory_file, allow_pickle=True).item()
    
    # Create save directory
    save_dir = "trajectory_visualizations"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create subdirectories for each type
    type_dirs = {}
    for i in range(4):
        type_name = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}[i]
        type_dir = os.path.join(save_dir, type_name)
        if not os.path.exists(type_dir):
            os.makedirs(type_dir)
        type_dirs[i] = type_dir
    
    # Visualize each trajectory
    print("Starting trajectory visualization...")
    for key, trajectory in tqdm(trajectories.items()):
        traj_type = trajectory['type']
        visualize_trajectory(trajectory, key, type_dirs[traj_type])
    
    print(f"Visualization complete! All images saved to {save_dir} directory")
    
    # Print statistics
    print("\nTrajectory statistics:")
    for i in range(4):
        type_name = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}[i]
        count = sum(1 for key in trajectories if f"type_{i}" in key)
        print(f"Type {type_name}: {count} trajectories")

if __name__ == "__main__":
    main()