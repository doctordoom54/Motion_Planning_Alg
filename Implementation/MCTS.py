import numpy as np
import sys
import os
import math
import random
from typing import List, Optional, Tuple

# Add parent directory to path to import obstacle course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obstacle_course.course_visualizer import CourseVisualizer

class MCTSNode:
    """
    Node for Monte Carlo Tree Search.
    Each node represents a state (position) in the 2D space.
    """
    def __init__(self, position: Tuple[float, float], parent=None, action=None):
        """
        Initialize MCTS node.
        
        Args:
            position (tuple): (x, y) position in 2D space
            parent (MCTSNode): Parent node in the tree
            action (tuple): Action taken to reach this node from parent
        """
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.action = action  # Action that led to this state
        
        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.children = []
        self.untried_actions = []  # Will be set by MCTS when needed
        
    def get_possible_actions(self, goal_pos: np.ndarray = None) -> List[Tuple[float, float]]:
        """
        Get all possible actions from current position.
        Actions are movement directions with fixed step size.
        
        Args:
            goal_pos (np.ndarray): Goal position for goal-biased action
        
        Returns:
            List of action tuples (dx, dy)
        """
        step_size = 3.0  # Reduced step size for better navigation
        actions = []
        
        # 8-directional movement (N, NE, E, SE, S, SW, W, NW)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            actions.append((dx, dy))
        
        # Add smaller step actions for better local navigation
        small_step = 1.0
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = small_step * np.cos(angle)
            dy = small_step * np.sin(angle)
            actions.append((dx, dy))
        
        # Add goal-biased action if goal is provided and close enough
        if goal_pos is not None:
            goal_direction = goal_pos - self.position
            goal_distance = np.linalg.norm(goal_direction)
            
            if goal_distance > 0:
                # If goal is within step size, add direct action to goal
                if goal_distance <= step_size * 1.5:
                    actions.append((goal_direction[0], goal_direction[1]))
                else:
                    # Add action towards goal with fixed step size
                    goal_unit = goal_direction / goal_distance
                    goal_action = goal_unit * step_size
                    actions.append((goal_action[0], goal_action[1]))
            
        return actions
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        # If untried_actions hasn't been initialized yet, it's not expanded
        if not hasattr(self, 'untried_actions'):
            return False
        return len(self.untried_actions) == 0 and len(self.children) > 0
    
    def is_terminal(self, goal_pos: np.ndarray, goal_tolerance: float) -> bool:
        """
        Check if this node represents a terminal state (goal reached).
        
        Args:
            goal_pos (np.ndarray): Goal position
            goal_tolerance (float): Distance tolerance to goal
            
        Returns:
            bool: True if goal is reached
        """
        return np.linalg.norm(self.position - goal_pos) <= goal_tolerance
    
    def ucb1_value(self, exploration_param: float = 1.414) -> float:
        """
        Calculate UCB1 (Upper Confidence Bound) value for node selection.
        
        Args:
            exploration_param (float): Exploration parameter (sqrt(2) is theoretical optimum)
            
        Returns:
            float: UCB1 value
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None:
            return self.total_reward / self.visits
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search implementation for 2D navigation.
    """
    def __init__(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float], 
                 course, max_iterations: int = 1000, exploration_param: float = 1.414):
        """
        Initialize MCTS planner.
        
        Args:
            start_pos (tuple): Starting position (x, y)
            goal_pos (tuple): Goal position (x, y)
            course: ObstacleCourse object for collision checking
            max_iterations (int): Maximum MCTS iterations
            exploration_param (float): UCB1 exploration parameter
        """
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.course = course
        self.max_iterations = max_iterations
        self.exploration_param = exploration_param
        self.goal_tolerance = 5.0  # Distance tolerance to reach goal
        
        # Initialize root node
        self.root = MCTSNode(start_pos)
        
    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Select a leaf node using UCB1.
        
        Args:
            node (MCTSNode): Current node to start selection from
            
        Returns:
            MCTSNode: Selected leaf node
        """
        while not node.is_terminal(self.goal_pos, self.goal_tolerance) and node.is_fully_expanded():
            # Check if node has children before selecting
            if not node.children:
                break
                
            # Select child with highest UCB1 value
            node = max(node.children, key=lambda child: child.ucb1_value(self.exploration_param))
        
        return node
    
    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expansion phase: Add a new child node for an untried action.
        
        Args:
            node (MCTSNode): Node to expand
            
        Returns:
            MCTSNode or None: New child node if expansion successful, None otherwise
        """
        if node.is_terminal(self.goal_pos, self.goal_tolerance):
            return None
        
        # Initialize untried actions if not done yet
        if not hasattr(node, 'untried_actions'):
            node.untried_actions = node.get_possible_actions(self.goal_pos)
        
        if not node.untried_actions:
            return None
        
        # Bias towards goal-directed action if available
        goal_direction = self.goal_pos - node.position
        goal_distance = np.linalg.norm(goal_direction)
        
        # Check if there's a goal-biased action in untried actions
        goal_biased_action = None
        if goal_distance > 0:
            for action in node.untried_actions:
                action_vec = np.array(action)
                # Check if this action points towards goal
                if np.dot(action_vec, goal_direction) > 0.7 * np.linalg.norm(action_vec) * goal_distance:
                    goal_biased_action = action
                    break
        
        # Select action (bias towards goal if available)
        if goal_biased_action is not None and random.random() < 0.6:  # 60% chance to pick goal-biased action
            action = goal_biased_action
        else:
            action = random.choice(node.untried_actions)
        
        node.untried_actions.remove(action)
        
        # Calculate new position
        new_position = node.position + np.array(action)
        
        # Check bounds
        if (new_position[0] < 0 or new_position[0] >= self.course.width or
            new_position[1] < 0 or new_position[1] >= self.course.height):
            return None
        
        # Check collision
        if not self.is_collision_free(node.position, new_position):
            return None
        
        # Create new child node
        child = MCTSNode(new_position, parent=node, action=action)
        node.children.append(child)
        
        return child
    
    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Run goal-biased rollout from given node to estimate value.
        
        Args:
            node (MCTSNode): Node to start simulation from
            
        Returns:
            float: Estimated reward/value
        """
        current_pos = node.position.copy()
        max_simulation_steps = 50
        step_size = 3.0  # Match the reduced step size
        
        for step in range(max_simulation_steps):
            # Check if goal is reached
            distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)
            if distance_to_goal <= self.goal_tolerance:
                # Goal reached - high reward based on how quickly we got there
                steps_reward = (max_simulation_steps - step) * 2.0
                distance_reward = (200.0 - np.linalg.norm(node.position - self.goal_pos)) * 0.5
                return 200.0 + steps_reward + distance_reward
            
            # Goal-biased action selection (80% towards goal, 20% random)
            if random.random() < 0.8:
                # Move towards goal
                goal_direction = self.goal_pos - current_pos
                goal_distance = np.linalg.norm(goal_direction)
                
                if goal_distance > 0:
                    if goal_distance <= step_size:
                        # Can reach goal directly
                        action = goal_direction
                    else:
                        # Move towards goal with step size
                        goal_unit = goal_direction / goal_distance
                        action = goal_unit * step_size
                else:
                    # At goal position
                    break
            else:
                # Random action
                angle = random.uniform(0, 2 * np.pi)
                action = np.array([step_size * np.cos(angle), step_size * np.sin(angle)])
            
            new_pos = current_pos + action
            
            # Check bounds
            if (new_pos[0] < 0 or new_pos[0] >= self.course.width or
                new_pos[1] < 0 or new_pos[1] >= self.course.height):
                break
            
            # Check collision
            if not self.is_collision_free(current_pos, new_pos):
                break
            
            current_pos = new_pos
        
        # Simulation ended without reaching goal - reward based on final distance to goal
        final_distance = np.linalg.norm(current_pos - self.goal_pos)
        initial_distance = np.linalg.norm(node.position - self.goal_pos)
        
        # Reward improvement in distance to goal
        distance_improvement = initial_distance - final_distance
        base_reward = max(0, distance_improvement * 10.0)  # Reward for getting closer
        
        # Additional reward based on how close we got
        max_distance = np.sqrt(self.course.width**2 + self.course.height**2)
        proximity_reward = max(0, (max_distance - final_distance) / max_distance * 50.0)
        
        return base_reward + proximity_reward
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update statistics of all nodes in path to root.
        
        Args:
            node (MCTSNode): Node to start backpropagation from
            reward (float): Reward to propagate
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """
        Check if the path from from_pos to to_pos is collision-free.
        
        Args:
            from_pos (np.ndarray): Starting position
            to_pos (np.ndarray): Ending position
            
        Returns:
            bool: True if collision-free, False otherwise
        """
        # Sample points along the line
        distance = np.linalg.norm(to_pos - from_pos)
        num_samples = max(5, int(distance * 2))  # More samples for better collision detection
        
        for i in range(num_samples + 1):
            t = i / num_samples if num_samples > 0 else 0
            point = from_pos + t * (to_pos - from_pos)
            
            # Ensure point is within bounds before checking obstacle
            if (point[0] < 0 or point[0] >= self.course.width or
                point[1] < 0 or point[1] >= self.course.height):
                return False
            
            # Check if point is in obstacle - handle floating point coordinates properly
            x_int = int(round(point[0]))
            y_int = int(round(point[1]))
            
            # Make sure rounded coordinates are still in bounds
            if (x_int < 0 or x_int >= self.course.width or
                y_int < 0 or y_int >= self.course.height):
                return False
                
            if self.course.point_in_obstacle(x_int, y_int):
                return False
                
        return True
    
    def search(self) -> Optional[MCTSNode]:
        """
        Main MCTS search algorithm.
        
        Returns:
            MCTSNode or None: Best leaf node found, None if no solution
        """
        print(f"Starting MCTS search with {self.max_iterations} iterations...")
        
        successful_expansions = 0
        failed_expansions = 0
        
        for iteration in range(self.max_iterations):
            if iteration % 100 == 0:
                print(f"MCTS Iteration {iteration}/{self.max_iterations}, Successful expansions: {successful_expansions}, Failed: {failed_expansions}")
            
            # 1. Selection
            leaf = self.select(self.root)
            
            # 2. Expansion
            if not leaf.is_terminal(self.goal_pos, self.goal_tolerance):
                child = self.expand(leaf)
                if child is not None:
                    leaf = child
                    successful_expansions += 1
                else:
                    failed_expansions += 1
            
            # 3. Simulation
            reward = self.simulate(leaf)
            
            # 4. Backpropagation
            self.backpropagate(leaf, reward)
            
            # Check if goal reached in tree
            if leaf.is_terminal(self.goal_pos, self.goal_tolerance):
                print(f"Goal reached in MCTS tree at iteration {iteration}!")
                return leaf
            
            # Also check if we're very close and can connect directly
            if iteration % 50 == 0:  # Check every 50 iterations
                best_node = self.find_closest_node_to_goal()
                if best_node and np.linalg.norm(best_node.position - self.goal_pos) <= self.goal_tolerance * 2:
                    # Try to connect directly to goal
                    if self.is_collision_free(best_node.position, self.goal_pos):
                        # Create goal node
                        goal_node = MCTSNode(self.goal_pos, parent=best_node, 
                                           action=self.goal_pos - best_node.position)
                        best_node.children.append(goal_node)
                        print(f"Direct connection to goal established at iteration {iteration}!")
                        return goal_node
        
        print(f"Search completed. Total successful expansions: {successful_expansions}, Failed: {failed_expansions}")
        # Find best path to goal or closest node
        best_node = self.find_best_path()
        return best_node
    
    def find_best_path(self) -> Optional[MCTSNode]:
        """
        Find the best path from root by following highest value nodes.
        
        Returns:
            MCTSNode: Best terminal node found
        """
        def find_best_leaf(node):
            if not node.children:
                return node
            
            # Find child with best average reward
            best_child = max(node.children, 
                           key=lambda child: child.total_reward / child.visits if child.visits > 0 else 0)
            return find_best_leaf(best_child)
        
        return find_best_leaf(self.root)
    
    def find_closest_node_to_goal(self) -> Optional[MCTSNode]:
        """
        Find the node in the tree closest to the goal.
        
        Returns:
            MCTSNode: Closest node to goal
        """
        def traverse_tree(node):
            nodes = [node]
            for child in node.children:
                nodes.extend(traverse_tree(child))
            return nodes
        
        all_nodes = traverse_tree(self.root)
        if not all_nodes:
            return None
        
        closest_node = min(all_nodes, 
                          key=lambda node: np.linalg.norm(node.position - self.goal_pos))
        return closest_node
    
    def extract_path(self, leaf_node: MCTSNode) -> List[MCTSNode]:
        """
        Extract path from root to given leaf node.
        
        Args:
            leaf_node (MCTSNode): Target leaf node
            
        Returns:
            List[MCTSNode]: Path from root to leaf
        """
        path = []
        current = leaf_node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path
    
    def plan(self) -> Optional[List[MCTSNode]]:
        """
        Execute MCTS planning and return path.
        
        Returns:
            List[MCTSNode] or None: Path from start to goal, None if no path found
        """
        best_leaf = self.search()
        
        if best_leaf is None:
            print("MCTS: No path found")
            return None
        
        path = self.extract_path(best_leaf)
        
        # Check if goal was actually reached
        if best_leaf.is_terminal(self.goal_pos, self.goal_tolerance):
            print(f"MCTS: Goal reached! Path length: {len(path)} nodes")
        else:
            final_distance = np.linalg.norm(best_leaf.position - self.goal_pos)
            print(f"MCTS: Best path found. Final distance to goal: {final_distance:.2f}")
        
        return path


def main():
    """
    Example usage of MCTS planner.
    """
    # Import here to avoid circular imports
    from obstacle_course.generate_course import ObstacleCourse
    
    # Create obstacle course
    course = ObstacleCourse(100, 200, 15, obstacle_size=12)
    course.generate_obstacles()
    course.set_start_and_goal()
    
    # Get start and goal from course
    start_pos = course.start
    goal_pos = course.goal
    
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    
    # Debug: Check if start position is valid
    print(f"Start position in obstacle: {course.point_in_obstacle(start_pos[0], start_pos[1])}")
    
    # Debug: Test a few simple moves from start
    test_node = MCTSNode(start_pos)
    test_actions = test_node.get_possible_actions(goal_pos)
    print(f"Number of possible actions: {len(test_actions)}")
    
    # Test first few actions
    for i, action in enumerate(test_actions[:5]):
        new_pos = np.array(start_pos) + np.array(action)
        in_bounds = (0 <= new_pos[0] < course.width and 0 <= new_pos[1] < course.height)
        collision_free = True
        if in_bounds:
            collision_free = not course.point_in_obstacle(int(new_pos[0]), int(new_pos[1]))
        print(f"Action {i}: {action} -> {new_pos}, in_bounds: {in_bounds}, collision_free: {collision_free}")
    
    # Create and run MCTS planner
    mcts_planner = MCTS(
        start_pos=start_pos,
        goal_pos=goal_pos,
        course=course,
        max_iterations=2000,
        exploration_param=1.414
    )
    
    # Plan path
    path = mcts_planner.plan()
    
    if path:
        print(f"MCTS found path with {len(path)} nodes")
        
        # Calculate path cost
        total_cost = 0
        for i in range(1, len(path)):
            segment_cost = np.linalg.norm(path[i].position - path[i-1].position)
            total_cost += segment_cost
        print(f"Total path cost: {total_cost:.2f}")
        
        # Visualization
        visualizer = CourseVisualizer(figsize=(12, 10))
        visualizer.plot_obstacles(
            obstacles=course.obstacles,
            width=course.width,
            height=course.height,
            start=start_pos,
            goal=goal_pos,
            title="MCTS Path Planning"
        )
        
        # Plot MCTS tree (sample of nodes for clarity)
        nodes_to_plot = [mcts_planner.root]
        queue = [mcts_planner.root]
        
        while queue and len(nodes_to_plot) < 500:  # Limit tree visualization
            node = queue.pop(0)
            for child in node.children:
                if child.visits > 5:  # Only plot well-visited nodes
                    nodes_to_plot.append(child)
                    queue.append(child)
        
        # Plot tree edges
        for node in nodes_to_plot:
            if node.parent is not None:
                x_vals = [node.parent.position[0], node.position[0]]
                y_vals = [node.parent.position[1], node.position[1]]
                visualizer.ax.plot(x_vals, y_vals, 'b-', alpha=0.3, linewidth=0.5)
        
        # Plot final path
        path_x = [node.position[0] for node in path]
        path_y = [node.position[1] for node in path]
        visualizer.ax.plot(path_x, path_y, 'r-', linewidth=3, label=f'MCTS Path (Cost: {total_cost:.1f})')
        visualizer.ax.legend()
        
        visualizer.show()
    else:
        print("No path found")


if __name__ == "__main__":
    main()