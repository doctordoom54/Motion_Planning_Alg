import numpy as np
import math
import random
from typing import List, Optional, Tuple
from obstacle_course.generate_course import ObstacleCourse


class MCTSNode:
    def __init__(self, position: Tuple[float, float], velocity: Tuple[float, float], 
                 acceleration: Tuple[float, float], parent=None, action=None):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.array(acceleration)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = False

    def ucb_value(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB value for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def add_child(self, child_node):
        """Add a child node"""
        child_node.parent = self
        self.children.append(child_node)

    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.total_reward += reward

    def is_expandable(self, max_children: int) -> bool:
        """Check if node has reached maximum children"""
        return len(self.children) <= max_children


class MCTSPlanner:
    def __init__(self, course: ObstacleCourse, start_pos: Tuple[float, float], 
                 goal_pos: Tuple[float, float], start_vel: Tuple[float, float] = (0.0, 0.0), 
                 start_acc: Tuple[float, float] = (0.0, 0.0), dt: float = 1.0, 
                 vmax: float = 30.0, amax: float = 15.0, amin: float = 15.0, 
                 max_iterations: int = 5000, rollout_horizon: int = 20, 
                 goal_tolerance: float = 2.0, goal_vel_tolerance: float = 0.5,
                 uct_c: float = 1.4, widen_k: float = 2.0,
                 widen_alpha: float = 0.5, direct_connect_radius: float = 10.0,
                  
                 
                 ):
        
        self.course = course
        self.start_pos = np.array(start_pos)
        self.start_vel = np.array(start_vel)
        self.start_acc = np.array(start_acc)
        self.goal_pos = np.array(goal_pos)
        self.dt = dt
        self.vmax = vmax
        self.amax = amax
        self.amin = -amin  # Make negative for lower bound
        self.max_iterations = max_iterations
        self.rollout_horizon = rollout_horizon
        self.goal_tolerance = goal_tolerance
        self.goal_vel_tolerance = goal_vel_tolerance  # Velocity must be near zero at goal
        
        # Progressive widening parameters
        self.k = widen_k
        self.alpha = widen_alpha
        
        # UCT parameter
        self.uct_c = uct_c
        
        # Reward structure
        self.reward_goal = 1000
        self.reward_collision = -1000
        self.reward_progress = 100
        
        # Direct connection parameters
        self.direct_connect_radius = direct_connect_radius
        
        
        
        
        
        # Root node
        self.root = MCTSNode(start_pos, start_vel, start_acc)
        self.best_goal_node = None

    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """Main MCTS planning loop"""
        for iteration in range(self.max_iterations):
            # Print progress every 500 iterations
            if (iteration + 1) % 10000 == 0:
                print(f"MCTS Progress: {iteration + 1}/{self.max_iterations} iterations completed")
            
            # Check terminal condition - direct connect if close to goal
            #if self._can_direct_connect(self.root):
            #    return self._finalize_direct_connect(self.root)
            
            # 1. Selection phase - traverse tree to find a node to expand
            leaf = self._select(self.root)
            
            # Every 100 iterations, check if selected leaf can connect to goal with any feasible action
            if (iteration + 1) % 50 == 0:
                
                if self._can_feasible_connect_to_goal(leaf):
                    print(f"✓ Direct connection found at iteration {iteration + 1}!")
                    return self._finalize_feasible_connect(leaf)
            
            # Check terminal condition from selected leaf
            if self._can_direct_connect(leaf):
                return self._finalize_direct_connect(leaf)
            
            # 2. Expansion phase - add new child if not fully expanded
            max_children = self._progressive_widening_limit(leaf)
            if leaf.is_expandable(max_children):
                child = self._expand(leaf)
                if child is not None:
                    # 3. Rollout phase - one random rollout from new child
                    reward = self._rollout(child)
                    # 4. Backpropagation phase - update path to root
                    self._backpropagate(child, reward)
                else:
                    # Failed to expand, give small penalty
                    self._backpropagate(leaf, -10)
            else:
                # Node is fully expanded, so we perform a rollout from its best child to refine scores
                if leaf.children:
                    print("stupid else block was triggered, leaf.children:", len(leaf.children))
                    best_child = max(leaf.children, key=lambda c: c.ucb_value(self.uct_c))
                    reward = self._rollout(best_child)
                    self._backpropagate(best_child, reward)
        
        # Return best path found
        return self._extract_path()

    def _progressive_widening_limit(self, node: MCTSNode) -> int:
        """Calculate the maximum number of children for progressive widening"""
        if node.visits == 0:
            return 1 # Allow at least one child for a new node
        return int(self.k * (node.visits ** self.alpha))

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase - traverse tree using UCB until a node that is not fully 
        expanded is found.
        """
        current_node = node
        
        while not current_node.is_terminal:
            max_children = self._progressive_widening_limit(current_node)
            
            # If the node is not fully expanded, we've found our target for expansion
            if current_node.is_expandable(max_children):
                return current_node
            
            # If the node is fully expanded but has no children, it's a dead end.
            # This can happen if max_children is 0 for a visited node. Return it.
            if not current_node.children:
                return current_node

            # Node is fully expanded, so use UCB to select the best child and continue descending
            best_child = max(current_node.children, 
                             key=lambda child: child.ucb_value(self.uct_c))
            current_node = best_child
            
        return current_node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expansion phase - add new child to the tree"""
        # Debugging counters
        bounds_failures = 0
        collision_failures = 0
        
        # Try multiple times to find a valid expansion
        for _ in range(500):
            # Sample a random action (acceleration)
            action = self._sample_action(node.position, node.velocity)
            
            # Integrate dynamics to get new state
            new_pos, new_vel, new_acc = self._integrate(node, action)
            
            # Check if new state is valid
            if not self._in_bounds(new_pos):
                bounds_failures += 1
                continue
            
            if not self._line_collision_free(node.position, new_pos):
                collision_failures += 1
                continue
            
            # Create new child node
            child = MCTSNode(new_pos, new_vel, new_acc, parent=node, action=action)
            
            # Check if child reaches goal
            if self._is_goal(child):
                child.is_terminal = True
                self.best_goal_node = child
            
            # Add child to parent
            node.add_child(child)
            return child
        
        # Failed to expand after multiple attempts - show debug info
        # (This part is unchanged)
        
        #print(f"Expansion failed after 500 attempts:")
        
            
        #print(f"  Position: {node.position}")
        #print(f"  Velocity: {node.velocity} (magnitude: {np.linalg.norm(node.velocity):.2f})")
        #print(f"  Bounds failures: {bounds_failures}/500")
        #print(f"  Collision failures: {collision_failures}/500")
        #print(f"  Course bounds: width={self.course.width}, height={self.course.height}")
        #print(f"  Parameters: dt={self.dt}, vmax={self.vmax}, amax={self.amax}")
        
        return None

    def _rollout(self, node: MCTSNode) -> float:
        """Rollout phase - simulate random actions to estimate node value"""
        if node.is_terminal:
            return self.reward_goal
        
        current_pos = node.position.copy()
        current_vel = node.velocity.copy()
        current_acc = node.acceleration.copy()
        
        total_reward = 0.0
        
        for step in range(self.rollout_horizon):
            # Sample random action
            action = self._sample_action(node.position, node.velocity)
            
            # Integrate one step
            next_pos, next_vel, next_acc = self._integrate_state(current_pos, current_vel, current_acc, action)
            
            # Check for collision
            if not self._in_bounds(next_pos) or not self._line_collision_free(current_pos, next_pos):
                return total_reward + self.reward_collision
            
            # Check for goal
            if np.linalg.norm(next_pos - self.goal_pos) <= self.goal_tolerance:
                return total_reward + self.reward_goal
            
            # Small reward for progress
            progress = np.linalg.norm(current_pos - self.goal_pos) - np.linalg.norm(next_pos - self.goal_pos)
            total_reward += progress * self.reward_progress
            
            # Update state
            current_pos = next_pos
            current_vel = next_vel
            current_acc = next_acc
        
        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase - update statistics up the tree"""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent #set current to its parent

    def _sample_action(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Sample an action with goal biasing and boundary avoidance heuristics.
        """
        # Sample magnitude as positive value (0 to amax)
        # This ensures each direction is distinct and we get uniform distribution
        magnitude = random.uniform(0, self.amax)

        # Define 4 diagonal directions: 45°, 135°, 225°, 315°
        diagonal_directions = [
            (1, 1),    # 45°   (northeast)
            (-1, 1),   # 135°  (northwest)
            (-1, -1),  # 225°  (southwest)
            (1, -1)    # 315°  (southeast)
        ]

        # Randomly select one of the 4 diagonal directions
        direction = random.choice(diagonal_directions)

        # Create action where |ax| = |ay| = magnitude
        # This gives us ax, ay in range [-amax, amax] with |ax| = |ay|
        ax = direction[0] * magnitude
        ay = direction[1] * magnitude

        action = np.array([ax, ay])

        return action

    def _integrate(self, node: MCTSNode, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate dynamics to get next state"""
        return self._integrate_state(node.position, node.velocity, node.acceleration, action)

    def _integrate_state(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, 
                         action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate dynamics from given state"""
        # New acceleration is the action
        new_acc = action.copy()
        
        # Update velocity: v = v + a * dt
        new_vel = vel + new_acc * self.dt
        
        # Clamp velocity to maximum
        vel_mag = np.linalg.norm(new_vel)
        if vel_mag > self.vmax:
            new_vel = new_vel * (self.vmax / vel_mag)
        
        # Update position: x = x + v * dt + 0.5 * a * dt^2
        new_pos = pos + vel * self.dt + 0.5 * new_acc * (self.dt ** 2)
        
        return new_pos, new_vel, new_acc

    def _in_bounds(self, pos: np.ndarray) -> bool:
        """Check if position is within course boundaries"""
        return 0 <= pos[0] <= self.course.width and 0 <= pos[1] <= self.course.height

    def _line_collision_free(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """Check if line segment is collision-free"""
        # Sample points along line segment
        distance = np.linalg.norm(p1 - p0)
        if distance == 0:
            return not self.course.point_in_obstacle(p0[0], p0[1])
        
        num_samples = max(2, int(distance / 0.5))
        for i in range(num_samples + 1):
            t = i / num_samples
            point = p0 + t * (p1 - p0)
            if self.course.point_in_obstacle(point[0], point[1]):
                return False
        
        return True

    def _is_goal(self, node: MCTSNode) -> bool:
        """Check if node satisfies goal conditions: position within tolerance AND velocity near zero"""
        pos_close = np.linalg.norm(node.position - self.goal_pos) <= self.goal_tolerance
        vel_small = np.linalg.norm(node.velocity) <= self.goal_vel_tolerance
        return pos_close and vel_small

    def _can_direct_connect(self, node: MCTSNode) -> bool:
        """Check if node can directly connect to goal"""
        distance = np.linalg.norm(node.position - self.goal_pos) #this is only checking for distance. 
        if distance > self.direct_connect_radius:
            return False
        
        return self._line_collision_free(node.position, self.goal_pos)

    def _finalize_direct_connect(self, node: MCTSNode) -> List[Tuple[float, float]]:
        """Create direct connection to goal with zero velocity"""
        # Create goal node with zero velocity and acceleration
        goal_node = MCTSNode(self.goal_pos, (0.0, 0.0), (0.0, 0.0), parent=node)
        goal_node.is_terminal = True
        node.add_child(goal_node)
        self.best_goal_node = goal_node
        
        return self._extract_path()
    
    def _can_feasible_connect_to_goal(self, node: MCTSNode) -> bool:
        """
        Check if node can connect to goal with zero final velocity using action within feasible bounds.
        Requires solving for action that brings robot to goal position with zero velocity.
        
        Using kinematic equations:
        goal_pos = pos + vel*dt + 0.5*action*dt^2
        0 = vel + action*dt  (final velocity should be zero)
        
        From second equation: action = -vel/dt
        Substituting into first: goal_pos = pos + vel*dt - 0.5*vel*dt = pos + 0.5*vel*dt
        This only works if goal_pos = pos + 0.5*vel*dt, which is too restrictive.
        
        Better approach: Check if we can reach goal with near-zero velocity in one step.
        """
        # Calculate action needed to bring velocity to zero
        # action_to_stop = -vel/dt
        action_to_stop = -node.velocity / self.dt
        
        # Calculate where we'd end up if we apply this action
        # new_pos = pos + vel*dt + 0.5*action*dt^2
        final_pos = node.position + node.velocity * self.dt + 0.5 * action_to_stop * (self.dt ** 2)
        final_vel = node.velocity + action_to_stop * self.dt  # Should be ~0
        
        # Check if final position is within goal tolerance
        distance_to_goal = np.linalg.norm(final_pos - self.goal_pos)
        if distance_to_goal > self.goal_tolerance:
            return False
        
        # Check if action to stop is within bounds
        action_magnitude = np.linalg.norm(action_to_stop)
        max_magnitude = max(abs(self.amin), abs(self.amax))
        if action_magnitude > max_magnitude:
            return False
        
        # Check if final velocity is small enough
        if np.linalg.norm(final_vel) > self.goal_vel_tolerance:
            return False
        
        # Check if path to final position is collision-free
        if not self._line_collision_free(node.position, final_pos):
            return False
        
        return True
    
    def _finalize_feasible_connect(self, node: MCTSNode) -> List[Tuple[float, float]]:
        """
        Create direct connection to goal with zero final velocity.
        Uses action that brings velocity to zero.
        """
        # Calculate action to bring velocity to zero
        action_to_stop = -node.velocity / self.dt
        
        # Integrate dynamics to get final state
        final_pos, final_vel, final_acc = self._integrate_state(
            node.position, node.velocity, node.acceleration, action_to_stop
        )
        
        # Create goal node with near-zero velocity
        goal_node = MCTSNode(final_pos, final_vel, final_acc, parent=node, action=action_to_stop)
        goal_node.is_terminal = True
        node.add_child(goal_node)
        self.best_goal_node = goal_node
        
        # Verify goal conditions
        distance_to_goal = np.linalg.norm(final_pos - self.goal_pos)
        final_vel_magnitude = np.linalg.norm(final_vel)
        print(f"  Final distance to goal: {distance_to_goal:.6f} (tolerance: {self.goal_tolerance})")
        print(f"  Final velocity magnitude: {final_vel_magnitude:.6f} (tolerance: {self.goal_vel_tolerance})")
        print(f"  Stopping action: {action_to_stop} (magnitude: {np.linalg.norm(action_to_stop):.3f})")
        
        return self._extract_path()

    def _extract_path(self) -> Optional[List[Tuple[float, float]]]:
        """Extract path from root to best goal node (positions only for backward compatibility)"""
        if self.best_goal_node is None:
            # Find the node closest to goal
            best_node = self._find_best_node(self.root)
            if best_node is None:
                return None
        else:
            best_node = self.best_goal_node
        
        # Traverse from best node back to root
        path = []
        current = best_node
        while current is not None:
            path.append((current.position[0], current.position[1]))
            current = current.parent
        
        # Reverse to get path from start to goal
        path.reverse()
        return path
    
    def extract_full_trace(self) -> Optional[List[dict]]:
        """
        Extract full trace with position, velocity, acceleration, and action at each node.
        Returns a list of dictionaries with complete state information.
        """
        if self.best_goal_node is None:
            # Find the node closest to goal
            best_node = self._find_best_node(self.root)
            if best_node is None:
                return None
        else:
            best_node = self.best_goal_node
        
        # Traverse from best node back to root
        trace = []
        current = best_node
        while current is not None:
            node_info = {
                'position': current.position.copy(),
                'velocity': current.velocity.copy(),
                'acceleration': current.acceleration.copy(),
                'action': current.action.copy() if current.action is not None else None,
                'visits': current.visits,
                'total_reward': current.total_reward
            }
            trace.append(node_info)
            current = current.parent
        
        # Reverse to get trace from start to goal
        trace.reverse()
        return trace

    def _find_best_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Find the node closest to goal in the tree using an iterative approach."""
        if not node:
            return None

        # Use a list as a stack for iterative depth-first search
        stack = [node]
        best_node = node
        best_distance = np.linalg.norm(node.position - self.goal_pos)

        while stack:
            current_node = stack.pop()
            
            # Check current node's distance
            distance = np.linalg.norm(current_node.position - self.goal_pos)
            if distance < best_distance:
                best_distance = distance
                best_node = current_node
            
            # Add children to the stack to visit them next
            for child in current_node.children:
                stack.append(child)
        
        return best_node