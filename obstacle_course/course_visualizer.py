import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union

class CourseVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Initialize the course visualizer.
        
        Args:
            figsize (tuple): Figure size for matplotlib (width, height)
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def initialize_plot(self):
        """Initialize a new matplotlib figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        
    def plot_grid(self, grid: np.ndarray, start: Optional[Tuple[int, int]] = None, 
                goal: Optional[Tuple[int, int]] = None, title: Optional[str] = None):
        """
        Plot the grid with obstacles, start, and goal points.
        
        Args:
            grid (np.ndarray): The grid to plot
            start (tuple, optional): Start point coordinates (x, y)
            goal (tuple, optional): Goal point coordinates (x, y)
            title (str, optional): Title for the plot
        """
        if self.fig is None:
            self.initialize_plot()
            
        self.ax.clear()
        
        # Plot the grid with obstacles
        self.ax.imshow(grid, cmap='binary')
        self.ax.grid(True, which='both', color='gray', linewidth=0.5)
        self.ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
        self.ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Plot start point if provided
        if start is not None:
            self.ax.plot(start[0], start[1], 'go', markersize=10, label='Start', 
                        markeredgecolor='white', markeredgewidth=2)
            
        # Plot goal point if provided
        if goal is not None:
            self.ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', 
                        markeredgecolor='white', markeredgewidth=2)
            
        if start is not None or goal is not None:
            self.ax.legend(loc='upper right')
        
        if title:
            self.ax.set_title(title)
    
    def show(self):
        """Display the plot."""
        if self.fig is not None:
            plt.show()
    
    def close(self):
        """Close the current figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
