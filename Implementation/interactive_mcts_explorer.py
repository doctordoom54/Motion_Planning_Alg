"""
Interactive MCTS Explorer
=========================
Run this script in IPython to explore MCTS tree structure, nodes, and decisions.

Usage:
    ipython
    >>> run interactive_mcts_explorer.py
    >>> # Explore the tree interactively
    >>> print_node_info(planner.root)
    >>> analyze_best_path(planner)
    >>> compare_children(planner.root)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from MCTS import MCTSPlanner


class MCTSAnalyzer:
    """Helper class to analyze MCTS tree structure and decisions"""
    
    def __init__(self, planner, course, path, full_trace):
        self.planner = planner
        self.course = course
        self.path = path
        self.full_trace = full_trace
        self.best_path_nodes = self._get_best_path_nodes()
    
    def _get_best_path_nodes(self):
        """Get the actual node objects in the best path"""
        if self.planner.best_goal_node is None:
            best_node = self.planner._find_best_node(self.planner.root)
        else:
            best_node = self.planner.best_goal_node
        
        nodes = []
        current = best_node
        while current is not None:
            nodes.append(current)
            current = current.parent
        nodes.reverse()
        return nodes
    
    def print_node_info(self, node, indent=0):
        """Print detailed information about a node"""
        prefix = "  " * indent
        print(f"{prefix}Node Info:")
        print(f"{prefix}  Position: {node.position}")
        print(f"{prefix}  Velocity: {node.velocity} (magnitude: {np.linalg.norm(node.velocity):.3f})")
        print(f"{prefix}  Acceleration: {node.acceleration}")
        print(f"{prefix}  Action (to reach this node): {node.action}")
        print(f"{prefix}  Visits: {node.visits}")
        print(f"{prefix}  Total Reward: {node.total_reward:.2f}")
        print(f"{prefix}  Avg Reward: {node.total_reward/node.visits if node.visits > 0 else 0:.2f}")
        print(f"{prefix}  Num Children: {len(node.children)}")
        print(f"{prefix}  Is Terminal: {node.is_terminal}")
        dist_to_goal = np.linalg.norm(node.position - self.planner.goal_pos)
        print(f"{prefix}  Distance to Goal: {dist_to_goal:.3f}")
    
    def print_children_info(self, node):
        """Print information about all children of a node with UCB scores"""
        print(f"\nNode at position {node.position} has {len(node.children)} children:")
        print("="*80)
        
        if len(node.children) == 0:
            print("No children (leaf node)")
            return
        
        for i, child in enumerate(node.children):
            ucb = child.ucb_value(self.planner.uct_c) if node.visits > 0 else float('inf')
            print(f"\nChild {i+1}:")
            print(f"  Position: {child.position}")
            print(f"  Action taken: {child.action}")
            print(f"  Velocity: {child.velocity} (mag: {np.linalg.norm(child.velocity):.3f})")
            print(f"  Visits: {child.visits}")
            print(f"  Total Reward: {child.total_reward:.2f}")
            print(f"  Avg Reward: {child.total_reward/child.visits if child.visits > 0 else 0:.2f}")
            print(f"  UCB Value: {ucb:.3f}")
            print(f"  Num Children: {len(child.children)}")
    
    def analyze_best_path(self):
        """Analyze the best path found by MCTS"""
        print("\n" + "="*80)
        print("BEST PATH ANALYSIS")
        print("="*80)
        print(f"Path length: {len(self.best_path_nodes)} nodes")
        print(f"Goal tolerance: {self.planner.goal_tolerance}")
        
        for i, node in enumerate(self.best_path_nodes):
            print(f"\n{'='*80}")
            print(f"Step {i} / {len(self.best_path_nodes)-1}")
            print(f"{'='*80}")
            self.print_node_info(node)
            
            if i < len(self.best_path_nodes) - 1:
                next_node = self.best_path_nodes[i+1]
                displacement = next_node.position - node.position
                print(f"  → Displacement to next: {displacement} (magnitude: {np.linalg.norm(displacement):.3f})")
    
    def compare_siblings(self, node_index_in_best_path):
        """Compare a node in the best path with its siblings (other children of same parent)"""
        if node_index_in_best_path == 0:
            print("Root node has no siblings")
            return
        
        node = self.best_path_nodes[node_index_in_best_path]
        parent = node.parent
        
        print(f"\n{'='*80}")
        print(f"COMPARING NODE AT STEP {node_index_in_best_path} WITH ITS SIBLINGS")
        print(f"{'='*80}")
        print(f"Parent position: {parent.position}")
        print(f"Parent has {len(parent.children)} children total")
        print(f"\nChosen child (in best path):")
        self.print_node_info(node, indent=1)
        
        print(f"\nAll siblings:")
        self.print_children_info(parent)
    
    def get_tree_statistics(self):
        """Get overall statistics about the MCTS tree"""
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        def get_max_depth(node, current_depth=0):
            if len(node.children) == 0:
                return current_depth
            return max(get_max_depth(child, current_depth + 1) for child in node.children)
        
        total_nodes = count_nodes(self.planner.root)
        max_depth = get_max_depth(self.planner.root)
        
        print("\n" + "="*80)
        print("MCTS TREE STATISTICS")
        print("="*80)
        print(f"Total nodes in tree: {total_nodes}")
        print(f"Maximum tree depth: {max_depth}")
        print(f"Best path length: {len(self.best_path_nodes)}")
        print(f"Root visits: {self.planner.root.visits}")
    
    def analyze_branching_by_depth(self, max_depth=None):
        """Analyze branching factor and node count at each depth level"""
        def collect_nodes_by_depth(node, depth=0, depth_dict=None):
            if depth_dict is None:
                depth_dict = {}
            
            if depth not in depth_dict:
                depth_dict[depth] = []
            depth_dict[depth].append(node)
            
            for child in node.children:
                collect_nodes_by_depth(child, depth + 1, depth_dict)
            
            return depth_dict
        
        depth_dict = collect_nodes_by_depth(self.planner.root)
        max_d = max(depth_dict.keys()) if max_depth is None else min(max_depth, max(depth_dict.keys()))
        
        print("\n" + "="*80)
        print("BRANCHING ANALYSIS BY DEPTH")
        print("="*80)
        print(f"{'Depth':<8} {'Nodes':<10} {'Avg Children':<15} {'Avg Visits':<15} {'Total Children':<15}")
        print("-" * 80)
        
        for depth in range(max_d + 1):
            if depth not in depth_dict:
                continue
            
            nodes_at_depth = depth_dict[depth]
            num_nodes = len(nodes_at_depth)
            avg_children = np.mean([len(n.children) for n in nodes_at_depth])
            avg_visits = np.mean([n.visits for n in nodes_at_depth])
            total_children = sum([len(n.children) for n in nodes_at_depth])
            
            print(f"{depth:<8} {num_nodes:<10} {avg_children:<15.2f} {avg_visits:<15.2f} {total_children:<15}")
        
        print("="*80)
        print("\nInterpretation:")
        print("- 'Nodes': Number of nodes at this depth level")
        print("- 'Avg Children': Average branching factor (children per node)")
        print("- 'Avg Visits': How often nodes at this depth were visited")
        print("- 'Total Children': Sum of all children at this depth")
        print("\nNote: Early depths should have higher visits and more children (progressive widening)")
    
    def get_best_path_data(self):
        """
        Extract all information from the best path for manipulation.
        
        Returns:
            dict: Dictionary containing arrays of:
                - positions: (N, 2) array of positions
                - velocities: (N, 2) array of velocities
                - accelerations: (N, 2) array of accelerations
                - actions: (N-1, 2) array of actions (one less than nodes)
                - visits: (N,) array of visit counts
                - rewards: (N,) array of total rewards
                - times: (N,) array of time steps (0, dt, 2*dt, ...)
        """
        N = len(self.best_path_nodes)
        
        positions = np.zeros((N, 2))
        velocities = np.zeros((N, 2))
        accelerations = np.zeros((N, 2))
        actions = np.zeros((N-1, 2))  # One less than nodes
        visits = np.zeros(N)
        rewards = np.zeros(N)
        times = np.arange(N) * self.planner.dt
        
        for i, node in enumerate(self.best_path_nodes):
            positions[i] = node.position
            velocities[i] = node.velocity
            accelerations[i] = node.acceleration
            visits[i] = node.visits
            rewards[i] = node.total_reward
            
            # Actions start from the second node (index 1)
            if i > 0 and node.action is not None:
                actions[i-1] = node.action
        
        # Calculate total action as sum of magnitudes of all actions
        total_action = 0.0
        for i in range(len(actions)):
            norm = np.linalg.norm(actions[i])
            total_action += norm

        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'actions': actions,
            'visits': visits,
            'rewards': rewards,
            'times': times,
            'num_nodes': N,
            'total_action': total_action
        }
    
    def get_best_path_positions(self):
        """Get just the positions from the best path as a (N, 2) numpy array"""
        return np.array([node.position for node in self.best_path_nodes])
    
    def get_best_path_velocities(self):
        """Get just the velocities from the best path as a (N, 2) numpy array"""
        return np.array([node.velocity for node in self.best_path_nodes])
    
    def get_best_path_accelerations(self):
        """Get just the accelerations from the best path as a (N, 2) numpy array"""
        return np.array([node.acceleration for node in self.best_path_nodes])
    
    def get_best_path_actions(self):
        """Get just the actions from the best path as a (N-1, 2) numpy array"""
        actions = []
        for i, node in enumerate(self.best_path_nodes):
            if i > 0 and node.action is not None:
                actions.append(node.action)
        return np.array(actions) if actions else np.array([]).reshape(0, 2)
    
    def plot_best_path_trajectory(self):
        """Plot position, velocity, and acceleration over time for the best path"""
        data = self.get_best_path_data()
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Position plots
        axes[0, 0].plot(data['times'], data['positions'][:, 0], 'b-', label='x', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position x')
        axes[0, 0].set_title('X Position over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(data['times'], data['positions'][:, 1], 'r-', label='y', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position y')
        axes[0, 1].set_title('Y Position over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Velocity plots
        vel_mag = np.linalg.norm(data['velocities'], axis=1)
        axes[1, 0].plot(data['times'], data['velocities'][:, 0], 'b-', label='vx', linewidth=2)
        axes[1, 0].plot(data['times'], data['velocities'][:, 1], 'r-', label='vy', linewidth=2)
        axes[1, 0].plot(data['times'], vel_mag, 'k--', label='|v|', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].set_title('Velocity Components over Time')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=self.planner.vmax, color='orange', linestyle='--', 
                          linewidth=1, label=f'vmax={self.planner.vmax}')
        
        axes[1, 1].plot(data['positions'][:, 0], data['positions'][:, 1], 'g-', linewidth=2)
        axes[1, 1].plot(data['positions'][0, 0], data['positions'][0, 1], 'go', 
                       markersize=10, label='Start')
        axes[1, 1].plot(data['positions'][-1, 0], data['positions'][-1, 1], 'ro', 
                       markersize=10, label='End')
        axes[1, 1].set_xlabel('X Position')
        axes[1, 1].set_ylabel('Y Position')
        axes[1, 1].set_title('2D Trajectory')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        # Acceleration plots
        acc_mag = np.linalg.norm(data['accelerations'], axis=1)
        axes[2, 0].plot(data['times'], data['accelerations'][:, 0], 'b-', label='ax', linewidth=2)
        axes[2, 0].plot(data['times'], data['accelerations'][:, 1], 'r-', label='ay', linewidth=2)
        axes[2, 0].plot(data['times'], acc_mag, 'k--', label='|a|', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Acceleration')
        axes[2, 0].set_title('Acceleration Components over Time')
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=self.planner.amax, color='orange', linestyle='--', 
                          linewidth=1, label=f'amax={self.planner.amax}')
        axes[2, 0].axhline(y=self.planner.amin, color='orange', linestyle='--', 
                          linewidth=1, label=f'amin={self.planner.amin}')
        
        # Actions plot (if available)
        if len(data['actions']) > 0:
            action_times = data['times'][1:]  # Actions correspond to transitions
            action_mag = np.linalg.norm(data['actions'], axis=1)
            axes[2, 1].plot(action_times, data['actions'][:, 0], 'b-', label='action_x', linewidth=2)
            axes[2, 1].plot(action_times, data['actions'][:, 1], 'r-', label='action_y', linewidth=2)
            axes[2, 1].plot(action_times, action_mag, 'k--', label='|action|', linewidth=2)
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Action')
            axes[2, 1].set_title('Actions over Time')
            axes[2, 1].legend(loc='upper right')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Best Path Trajectory Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_all_actions(self):
        """Extract all actions taken in the entire MCTS tree"""
        all_actions = []
        
        def collect_actions(node):
            """Recursively collect actions from all nodes"""
            for child in node.children:
                if child.action is not None:
                    action_info = {
                        'action': child.action.copy(),
                        'parent_position': node.position.copy(),
                        'child_position': child.position.copy(),
                        'parent_velocity': node.velocity.copy(),
                        'child_velocity': child.velocity.copy(),
                        'child_visits': child.visits,
                        'child_reward': child.total_reward,
                        'ucb_value': child.ucb_value(self.planner.uct_c) if node.visits > 0 else float('inf')
                    }
                    all_actions.append(action_info)
                # Recursively collect from children
                collect_actions(child)
        
        # Start collection from root
        collect_actions(self.planner.root)
        return all_actions
    
    def analyze_all_actions(self):
        """Analyze and print statistics about all actions in the tree"""
        all_actions = self.get_all_actions()
        
        print("\n" + "="*80)
        print("ALL ACTIONS IN MCTS TREE")
        print("="*80)
        print(f"Total actions (edges) in tree: {len(all_actions)}")
        
        if len(all_actions) == 0:
            print("No actions found in tree")
            return all_actions
        
        # Extract action arrays
        actions_array = np.array([a['action'] for a in all_actions])
        
        # Compute statistics
        action_magnitudes = np.linalg.norm(actions_array, axis=1)
        ax_values = actions_array[:, 0]
        ay_values = actions_array[:, 1]
        
        print(f"\nAction Magnitude Statistics:")
        print(f"  Min: {np.min(action_magnitudes):.3f}")
        print(f"  Max: {np.max(action_magnitudes):.3f}")
        print(f"  Mean: {np.mean(action_magnitudes):.3f}")
        print(f"  Std: {np.std(action_magnitudes):.3f}")
        
        print(f"\nAction Component (ax) Statistics:")
        print(f"  Min: {np.min(ax_values):.3f}")
        print(f"  Max: {np.max(ax_values):.3f}")
        print(f"  Mean: {np.mean(ax_values):.3f}")
        
        print(f"\nAction Component (ay) Statistics:")
        print(f"  Min: {np.min(ay_values):.3f}")
        print(f"  Max: {np.max(ay_values):.3f}")
        print(f"  Mean: {np.mean(ay_values):.3f}")
        
        # Check for diagonal constraint (|ax| = |ay|)
        abs_diff = np.abs(np.abs(ax_values) - np.abs(ay_values))
        print(f"\nDiagonal Constraint Check (|ax| - |ay|):")
        print(f"  Max deviation: {np.max(abs_diff):.6f}")
        print(f"  Mean deviation: {np.mean(abs_diff):.6f}")
        
        # Count action directions
        directions = {}
        for action in actions_array:
            ax, ay = action
            # Classify direction
            if ax > 0 and ay > 0:
                direction = "NE (45°)"
            elif ax < 0 and ay > 0:
                direction = "NW (135°)"
            elif ax < 0 and ay < 0:
                direction = "SW (225°)"
            elif ax > 0 and ay < 0:
                direction = "SE (315°)"
            else:
                direction = "Zero"
            
            directions[direction] = directions.get(direction, 0) + 1
        
        print(f"\nAction Direction Distribution:")
        for direction, count in sorted(directions.items()):
            percentage = (count / len(all_actions)) * 100
            print(f"  {direction}: {count} ({percentage:.1f}%)")
        
        return all_actions
    
    def print_all_actions(self, limit=None):
        """Print detailed information about all actions (optionally limited)"""
        all_actions = self.get_all_actions()
        
        print("\n" + "="*80)
        print("DETAILED ACTION LIST")
        print("="*80)
        
        num_to_print = len(all_actions) if limit is None else min(limit, len(all_actions))
        print(f"Showing {num_to_print} of {len(all_actions)} total actions")
        print("="*80)
        
        for i, action_info in enumerate(all_actions[:num_to_print]):
            print(f"\nAction {i+1}:")
            print(f"  Action: {action_info['action']}")
            print(f"  Magnitude: {np.linalg.norm(action_info['action']):.3f}")
            print(f"  Parent Position: {action_info['parent_position']}")
            print(f"  Child Position: {action_info['child_position']}")
            print(f"  Parent Velocity: {action_info['parent_velocity']}")
            print(f"  Child Velocity: {action_info['child_velocity']}")
            print(f"  Child Visits: {action_info['child_visits']}")
            print(f"  Child Total Reward: {action_info['child_reward']:.2f}")
            print(f"  UCB Value: {action_info['ucb_value']:.3f}")
        
        if limit and len(all_actions) > limit:
            print(f"\n... and {len(all_actions) - limit} more actions")
        
        return all_actions
        print(f"Root children: {len(self.planner.root.children)}")
        print(f"Branching factor (avg): {total_nodes / max_depth if max_depth > 0 else 0:.2f}")
    
    def visualize_tree_with_stats(self):
        """Create visualization with tree exploration and best path"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Full tree exploration
        visualizer1 = CourseVisualizer(figsize=(10, 10))
        visualizer1.fig = fig
        visualizer1.ax = ax1
        visualizer1.plot_obstacles(
            obstacles=self.course.obstacles,
            width=self.course.width,
            height=self.course.height,
            start=self.course.start,
            goal=self.course.goal,
            title="MCTS Tree Exploration"
        )
        
        # Plot all tree edges
        def plot_tree(node):
            for child in node.children:
                ax1.plot(
                    [node.position[0], child.position[0]], 
                    [node.position[1], child.position[1]], 
                    'k-', linewidth=0.3, alpha=0.4
                )
                plot_tree(child)
        
        plot_tree(self.planner.root)
        
        # Plot best path
        if self.path and len(self.path) > 1:
            px, py = zip(*self.path)
            ax1.plot(px, py, 'r-', linewidth=2, label='Best Path', zorder=10)
            ax1.legend(loc='upper right')
        
        # Right plot: Best path with visit counts
        visualizer2 = CourseVisualizer(figsize=(10, 10))
        visualizer2.fig = fig
        visualizer2.ax = ax2
        visualizer2.plot_obstacles(
            obstacles=self.course.obstacles,
            width=self.course.width,
            height=self.course.height,
            start=self.course.start,
            goal=self.course.goal,
            title="Best Path with Node Visits"
        )
        
        # Plot best path with visit information
        if self.path and len(self.path) > 1:
            px, py = zip(*self.path)
            ax2.plot(px, py, 'r-', linewidth=2, label='Best Path', zorder=10)
            
            # Add visit count annotations
            for i, node in enumerate(self.best_path_nodes):
                ax2.plot(node.position[0], node.position[1], 'ro', markersize=8, zorder=11)
                ax2.annotate(
                    f'v:{node.visits}',
                    (node.position[0], node.position[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_path(self):
        """Plot the MCTS tree exploration and the best path found"""
        visualizer = CourseVisualizer(figsize=(10, 10))
        visualizer.plot_obstacles(
            obstacles=self.course.obstacles,
            width=self.course.width,
            height=self.course.height,
            start=self.course.start,
            goal=self.course.goal,
            title="MCTS Path Planning"
        )
        
        # Plot all explored paths (MCTS tree) with thin black lines
        def plot_tree(node):
            """Recursively plot all edges in the MCTS tree"""
            for child in node.children:
                # Draw edge from parent to child
                visualizer.ax.plot(
                    [node.position[0], child.position[0]], 
                    [node.position[1], child.position[1]], 
                    'k-', linewidth=0.3, alpha=0.4
                )
                # Recursively plot children
                plot_tree(child)
        
        # Plot the entire tree starting from root
        print("Plotting MCTS tree exploration...")
        plot_tree(self.planner.root)
        
        # Plot the best path on top in red
        if self.path and len(self.path) > 1:
            px, py = zip(*self.path)
            visualizer.ax.plot(px, py, 'r-', linewidth=2, label='Best Path', zorder=10)
            visualizer.ax.legend(loc='upper right')
            print(f"Path length: {len(self.path)} nodes")
        else:
            print("No path to plot")
        
        plt.show()
    
    def plot_action_histogram(self, bins=30, range_min=-8, range_max=8):
        """Plot histogram of all actions taken in the tree"""
        all_actions = self.get_all_actions()
        
        if len(all_actions) == 0:
            print("No actions found in tree")
            return
        
        # Extract action components
        actions_array = np.array([a['action'] for a in all_actions])
        ax_values = actions_array[:, 0]
        ay_values = actions_array[:, 1]
        action_magnitudes = np.linalg.norm(actions_array, axis=1)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: ax (x-component) histogram
        axes[0, 0].hist(ax_values, bins=bins, range=(range_min, range_max), 
                        color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('ax (x-component)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title(f'Action X-Component Distribution\n(Total: {len(ax_values)} actions)', fontsize=14)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='upper right')
        
        # Subplot 2: ay (y-component) histogram
        axes[0, 1].hist(ay_values, bins=bins, range=(range_min, range_max), 
                        color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('ay (y-component)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title(f'Action Y-Component Distribution\n(Total: {len(ay_values)} actions)', fontsize=14)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='upper right')
        
        # Subplot 3: Action magnitude histogram
        mag_range_max = np.sqrt(2) * range_max  # Maximum magnitude for diagonal actions
        axes[1, 0].hist(action_magnitudes, bins=bins, range=(0, mag_range_max), 
                        color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Action Magnitude', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title(f'Action Magnitude Distribution\n(Mean: {np.mean(action_magnitudes):.3f})', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: 2D scatter plot of actions (ax vs ay)
        axes[1, 1].scatter(ax_values, ay_values, alpha=0.5, s=20, c='orange', edgecolor='black', linewidth=0.5)
        axes[1, 1].set_xlabel('ax (x-component)', fontsize=12)
        axes[1, 1].set_ylabel('ay (y-component)', fontsize=12)
        axes[1, 1].set_title('Action Space Coverage (ax vs ay)', fontsize=14)
        axes[1, 1].set_xlim(range_min, range_max)
        axes[1, 1].set_ylim(range_min, range_max)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add diagonal lines to show constraint
        diag_range = np.linspace(range_min, range_max, 100)
        axes[1, 1].plot(diag_range, diag_range, 'b--', linewidth=1, alpha=0.3, label='45°')
        axes[1, 1].plot(diag_range, -diag_range, 'b--', linewidth=1, alpha=0.3, label='135°')
        axes[1, 1].legend(loc='upper right')
        
        plt.suptitle('MCTS Action Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n" + "="*60)
        print("ACTION HISTOGRAM STATISTICS")
        print("="*60)
        print(f"Total actions plotted: {len(all_actions)}")
        print(f"\nax (x-component):")
        print(f"  Range: [{range_min}, {range_max}]")
        print(f"  Min: {np.min(ax_values):.3f}, Max: {np.max(ax_values):.3f}")
        print(f"  Mean: {np.mean(ax_values):.3f}, Std: {np.std(ax_values):.3f}")
        print(f"\nay (y-component):")
        print(f"  Range: [{range_min}, {range_max}]")
        print(f"  Min: {np.min(ay_values):.3f}, Max: {np.max(ay_values):.3f}")
        print(f"  Mean: {np.mean(ay_values):.3f}, Std: {np.std(ay_values):.3f}")
        print(f"\nMagnitude:")
        print(f"  Min: {np.min(action_magnitudes):.3f}, Max: {np.max(action_magnitudes):.3f}")
        print(f"  Mean: {np.mean(action_magnitudes):.3f}, Std: {np.std(action_magnitudes):.3f}")
        print("="*60)
        
        plt.tight_layout()
        plt.show()


def run_mcts_interactive():
    """Run MCTS and return analyzer for interactive exploration"""
    print("="*80)
    print("INTERACTIVE MCTS EXPLORER")
    print("="*80)
    
    # Setup obstacle course
    width = 60
    height = 60
    num_obstacles = 8
    obstacle_size = 8

    course = ObstacleCourse(n=width, m=height, num_obstacles=num_obstacles, obstacle_size=obstacle_size)
    course.generate_obstacles()
    course.set_start_and_goal()

    print(f"Start: {course.start}, Goal: {course.goal}")
    print(f"Distance: {np.linalg.norm(np.array(course.goal) - np.array(course.start)):.2f}")

    # Setup MCTS planner
    start_vel = (0.0, 0.0)
    start_acc = (0.0, 0.0)

    planner = MCTSPlanner(
        course=course,
        start_pos=course.start,
        goal_pos=course.goal,
        start_vel=start_vel,
        start_acc=start_acc,
        dt=0.4,
        vmax=35.50,
        amax=7,
        amin=-7,
        goal_tolerance=1,
        uct_c=np.sqrt(2),
        widen_k=6,
        widen_alpha=0.2,
        rollout_horizon=25,
        max_iterations=int(1e5),
        direct_connect_radius=5.0,
        
    )

    planner.reward_goal = 100.0
    planner.reward_collision = -100.0
    planner.reward_progress = 50

    # Run planning
    print("\nRunning MCTS planner...")
    path = planner.plan()
    planner.final_path = path
    
    # Extract full trace
    full_trace = planner.extract_full_trace()
    planner.full_trace = full_trace

    # Create analyzer
    analyzer = MCTSAnalyzer(planner, course, path, full_trace)
    
    # Print initial statistics
    analyzer.get_tree_statistics()
    
    print("\n" + "="*80)
    print("INTERACTIVE EXPLORATION READY!")
    print("="*80)
    print("\nAvailable objects:")
    print("  planner  - MCTSPlanner instance")
    print("  course   - ObstacleCourse instance")
    print("  path     - Best path (list of position tuples)")
    print("  analyzer - MCTSAnalyzer for interactive exploration")
    print("\nAvailable functions:")
    print("  analyzer.print_node_info(node)           - Print detailed node info")
    print("  analyzer.print_children_info(node)       - Print all children with UCB scores")
    print("  analyzer.analyze_best_path()             - Analyze entire best path")
    print("  analyzer.compare_siblings(step_index)    - Compare with siblings at step")
    print("  analyzer.get_tree_statistics()           - Print tree statistics")
    print("  analyzer.visualize_tree_with_stats()     - Create visualization")
    print("\n  ** NEW: Action Analysis **")
    print("  analyzer.get_all_actions()               - Get list of all actions in tree")
    print("  analyzer.analyze_all_actions()           - Print statistics about all actions")
    print("  analyzer.print_all_actions(limit=20)     - Print detailed action info (limit optional)")
    print("  analyzer.plot_action_histogram()         - Plot histogram of actions (range -3 to 3)")
    print("\nExamples:")
    print("  # Explore root node")
    print("  analyzer.print_node_info(planner.root)")
    print("  analyzer.print_children_info(planner.root)")
    print("\n  # Analyze best path")
    print("  analyzer.analyze_best_path()")
    print("\n  # Compare step 5 with its siblings")
    print("  analyzer.compare_siblings(5)")
    print("\n  # Get all actions in the tree")
    print("  all_actions = analyzer.get_all_actions()")
    print("  print(f'Total actions: {len(all_actions)}')")
    print("\n  # Analyze action statistics")
    print("  analyzer.analyze_all_actions()")
    print("\n  # Plot action histogram")
    print("  analyzer.plot_action_histogram()")
    print("  analyzer.plot_action_histogram(bins=50, range_min=-5, range_max=5)")
    print("\n  # Print first 20 actions in detail")
    print("  analyzer.print_all_actions(limit=20)")
    print("\n  # Visualize")
    print("  analyzer.visualize_tree_with_stats()")
    print("="*80)
    
    return planner, course, path, analyzer


# Run the interactive session
if __name__ == "__main__":
    planner, course, path, analyzer = run_mcts_interactive()
