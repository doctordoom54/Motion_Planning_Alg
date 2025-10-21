import numpy as np
import sys
import os

# Add parent directory to path to import obstacle course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obstacle_course.course_visualizer import CourseVisualizer

class KNode:
    def __init__(self, position, velocity, acceleration, parent=None):
        """
        Node for kinodynamic RRT tree.
        
        Args:
            position (np.array): [x, y] position
            velocity (np.array): [vx, vy] velocity
            acceleration (np.array): [ax, ay] acceleration
            parent (KNode): Parent node in the tree
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array(acceleration, dtype=float)
        self.parent = parent
        self.cost = 0.0  # Cost from start to this node

class KinodynamicRRT:
    def __init__(self, start_pos, start_vel, start_acc, goal_pos, course, 
                 max_iterations=None, step_size=None, goal_tolerance=None):
        """
        Kinodynamic RRT planner.
        
        Args:
            start_pos (list): [x, y] starting position
            start_vel (list): [vx, vy] starting velocity
            start_acc (list): [ax, ay] starting acceleration
            goal_pos (list): [x, y] goal position
            course (ObstacleCourse): Obstacle course object
            max_iterations (int): Maximum number of RRT iterations
            step_size (float): Step size for integration
            goal_tolerance (float): Distance tolerance to reach goal
        """
        self.start_node = KNode(start_pos, start_vel, start_acc)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.course = course
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        
        # Tree storage
        self.nodes = [self.start_node]
        self.goal_node = None
        
        # Control limits
        self.max_acceleration = 5.0  # Maximum acceleration magnitude
        self.max_velocity = 35.0     # Maximum velocity magnitude
        
        # Adapt parameters based on initial conditions
        self._adapt_parameters()
        
    def _adapt_parameters(self):
        """
        Adapt algorithm parameters based on initial conditions for better robustness.
        """
        initial_vel_mag = np.linalg.norm(self.start_node.velocity)
        initial_acc_mag = np.linalg.norm(self.start_node.acceleration)
        
        # Adapt step size based on initial velocity
        if initial_vel_mag > self.max_velocity * 0.5:
            self.step_size = min(self.step_size, 0.5)  # Smaller steps for high velocities
            print(f"Adapted step size to {self.step_size} due to high initial velocity")
        
        # Adapt goal tolerance based on initial dynamic state
        if initial_vel_mag > self.max_velocity * 0.7:
            self.goal_tolerance = max(self.goal_tolerance, 2.0)  # Larger tolerance for high velocities
            print(f"Adapted goal tolerance to {self.goal_tolerance} due to high initial velocity")
        
        # Adapt max iterations based on problem complexity
        problem_scale = np.sqrt(self.course.width * self.course.height)
        if initial_vel_mag > self.max_velocity * 0.5 or problem_scale > 50:
            self.max_iterations = max(self.max_iterations, 8000)#TODO change this back to 8000
            print(f"Increased max iterations to {self.max_iterations} for complex initial conditions")
        
    def sample_random_state(self):
        """
        Sample a random state with bias toward reachable configurations.
        
        Returns:
            tuple: (position, velocity, acceleration)
        """
        # Sample random position within course bounds
        pos = np.array([
            np.random.uniform(0, self.course.width),
            np.random.uniform(0, self.course.height)
        ])
        
        # Bias velocity sampling toward lower speeds for better reachability
        if np.random.random() < 0.5:  # 70% bias toward lower speeds
            vel_mag = np.random.exponential(self.max_velocity * 0.2)
            vel_mag = min(vel_mag, self.max_velocity)
        else:
            vel_mag = np.random.uniform(0, self.max_velocity)
        #direction of velocity 
        vel_angle = np.random.uniform(0, 2 * np.pi)
        vel = vel_mag * np.array([np.cos(vel_angle), np.sin(vel_angle)])
        
        
        
        return pos, vel
    
    def find_nearest_node(self, target_pos, target_vel=None):
        """
        Find the nearest node in the tree considering both position and velocity.
        
        Args:
            target_pos (np.array): Target position [x, y]
            target_vel (np.array, optional): Target velocity [vx, vy]
            
        Returns:
            KNode: Nearest node in the tree
        """
        min_cost = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            # Weighted distance considering position and velocity
            pos_dist = np.linalg.norm(node.position - target_pos)
            
            if target_vel is not None:
                vel_dist = np.linalg.norm(node.velocity - target_vel)
                # Weight position more heavily than velocity
                total_cost = pos_dist + vel_dist * self.step_size #selection of weight? comprehensive factor for vel
            else:
                total_cost = pos_dist
            
            if total_cost < min_cost:
                min_cost = total_cost 
                nearest_node = node
                
        return nearest_node
    
    def find_nearest_node_by_position(self, target_pos):
        """
        Find the nearest node in the tree considering only position (ignoring velocity).
        
        Args:
            target_pos (np.array): Target position [x, y]
            
        Returns:
            KNode: Nearest node in the tree by position only
        """
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            pos_dist = np.linalg.norm(node.position - target_pos)
            
            if pos_dist < min_dist:
                min_dist = pos_dist
                nearest_node = node
                
        return nearest_node
    
    def try_connect_to_goal(self, from_node):
        """
        Try to connect a node directly to the goal by calculating required action.
        Checks if action is within bounds, collision-free, and final velocity is valid.
        
        Args:
            from_node (KNode): Starting node to connect from
            
        Returns:
            KNode or None: Goal node if connection successful, None otherwise
        """
        # Calculate required displacement
        displacement = self.goal_pos - from_node.position
        distance = np.linalg.norm(displacement)
        
        if distance < 1e-6: #can we change this to 1? 
            # this is already goal
            if np.linalg.norm(from_node.velocity) <= self.max_velocity:
                goal_node = KNode(self.goal_pos, from_node.velocity, np.array([0.0, 0.0]), parent=from_node)
                return goal_node
            return None
        
        # Calculate required acceleration to reach goal in one step
        # Using kinematic equation: s = v0*t + 0.5*a*t^2
        # And: v_final = v0 + a*t
        # Where t = step_size
        t = self.step_size
        v0 = from_node.velocity
        
        # Solve for acceleration: a = 2*(s - v0*t) / t^2
        required_acc = 2.0 * (displacement - v0 * t) / (t * t)
        
        # Check if required acceleration is within bounds
        acc_mag = np.linalg.norm(required_acc)
        if acc_mag > self.max_acceleration:
            return None  # Action exceeds max acceleration
        
        # Calculate final velocity
        final_velocity = v0 + required_acc * t
        final_vel_mag = np.linalg.norm(final_velocity)
        
        # Check if final velocity is within bounds
        if final_vel_mag > self.max_velocity:
            return None  # Final velocity exceeds max velocity
        
        # Check for collision along the path
        if not self.check_collision(from_node.position, self.goal_pos):
            return None  # Collision detected
        
        # All checks passed - create goal node
        goal_node = KNode(self.goal_pos, final_velocity, required_acc, parent=from_node)
        goal_node.cost = from_node.cost + distance
        
        return goal_node
    
    def integrate_dynamics(self, node, control_acc, time_step):
        """
        Integrate system dynamics using first-order integration.
        
        Args:
            node (KNode): Current node
            control_acc (np.array): Control acceleration [ax, ay]
            time_step (float): Integration time step
            
        Returns:
            tuple: (new_position, new_velocity, new_acceleration)
        """
        # Limit control acceleration
        acc_mag = np.linalg.norm(control_acc)
        if acc_mag > self.max_acceleration:
            control_acc = control_acc * (self.max_acceleration / acc_mag) #make the control acceleration within limit
        
        # First-order integration: v = v0 + a*dt, x = x0 + v*dt
        new_velocity = node.velocity + control_acc * time_step
        
        # Limit velocity
        vel_mag = np.linalg.norm(new_velocity)
        if vel_mag > self.max_velocity:
            new_velocity = new_velocity * (self.max_velocity / vel_mag) #limit the velocity within max limit
        
        new_position = node.position + new_velocity * time_step
        new_acceleration = control_acc #new acceleration at the node
        
        return new_position, new_velocity, new_acceleration
    
    def steer_to_goal_with_stop(self, from_node, target_pos):
        """
        Steer toward goal while trying to achieve zero velocity at the target.
        this method is only called every 10th iteration to try to reach to goal directly from
        the current point in the tree
        Args:
            from_node (KNode): Starting node
            target_pos (np.array): Target position, also a node
            
        Returns:
            KNode or None: New node if steering is successful, None otherwise
        """
        direction = target_pos - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return None
        
        direction_unit = direction / distance
        current_vel = from_node.velocity
        
        # Calculate required deceleration to stop at goal
        # Using kinematic equation: v^2 = u^2 + 2*a*s
        # For final velocity = 0: a = -u^2 / (2*s)
        vel_toward_goal = np.dot(current_vel, direction_unit) # compoenent of velocity in the direction of goal for the state
        
        if distance > 0.1 and vel_toward_goal > 0:
            # Calculate deceleration needed to stop at goal
            required_decel = -(vel_toward_goal ** 2) / (2 * distance) #acceleration based on velocity and distance
            desired_acc = direction_unit * max(required_decel, -self.max_acceleration) #update decelration can be different than acceleration
        else:
            # Close to goal or moving away, gentle approach
            desired_acc = direction_unit * self.max_acceleration * 0.3 #parameter taken randomly again
        
        # Integrate dynamics
        new_pos, new_vel, new_acc = self.integrate_dynamics(
            from_node, desired_acc, self.step_size
        )
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.course.width or
            new_pos[1] < 0 or new_pos[1] >= self.course.height):
            return None
        
        new_node = KNode(new_pos, new_vel, new_acc, parent=from_node)
        new_node.cost = from_node.cost + np.linalg.norm(new_pos - from_node.position) #new cost is just based on position mag
        
        return new_node
    
    def steer(self, from_node, target_pos):
        """
        Steer from a node towards a target position with intelligent control.
        
        Args:
            from_node (KNode): Starting node
            target_pos (np.array): Target position
            
        Returns:
            KNode or None: New node if steering is successful, None otherwise
        """
        # Calculate direction to target
        direction = target_pos - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # Too close
            return None
        
        direction_unit = direction / distance
        
        # Intelligent control strategy
        # Consider current velocity and required deceleration
        current_vel = from_node.velocity
        vel_toward_target = np.dot(current_vel, direction_unit)
        
        # Adaptive step size based on velocity
        adaptive_step = max(0.1, min(self.step_size, self.step_size * (1.0 + np.linalg.norm(current_vel) / 10.0)))
        
        # Control strategy based on current state
        if vel_toward_target > 0 and distance < vel_toward_target * adaptive_step:
            # Need to decelerate to avoid overshooting
            desired_acc = -direction_unit * self.max_acceleration * 0.8
        else:
            # Accelerate toward target, but consider current velocity
            vel_component_perpendicular = current_vel - vel_toward_target * direction_unit
            damping_factor = max(0.3, 1.0 - np.linalg.norm(vel_component_perpendicular) / self.max_velocity)
            desired_acc = direction_unit * self.max_acceleration * damping_factor
        
        # Integrate dynamics with adaptive step size this gives a physical set of state
        new_pos, new_vel, new_acc = self.integrate_dynamics(
            from_node, desired_acc, adaptive_step
        )
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.course.width or
            new_pos[1] < 0 or new_pos[1] >= self.course.height):
            return None
        
        # Create new node
        new_node = KNode(new_pos, new_vel, new_acc, parent=from_node)
        new_node.cost = from_node.cost + np.linalg.norm(new_pos - from_node.position)
        
        return new_node
    
    def check_collision(self, from_pos, to_pos):
        """
        Check if the straight line path from from_pos to to_pos collides with obstacles.
        
        Args:
            from_pos (np.array): Starting position
            to_pos (np.array): Ending position
            
        Returns:
            bool: True if collision-free, False if collision detected
        """
        # Sample points along the line
        num_samples = max(10, int(np.linalg.norm(to_pos - from_pos)))
        for i in range(num_samples + 1):
            t = i / num_samples
            point = from_pos + t * (to_pos - from_pos)
            
            # Check if point is in obstacle
            if self.course.point_in_obstacle(int(point[0]), int(point[1])): #taking this method from course object 
                return False
                
        return True
    
    def is_goal_reached(self, node):
        """
        Check if the goal is reached within tolerance.
        
        Args:
            node (KNode): Node to check
            
        Returns:
            bool: True if goal is reached
        """
        position_close = np.linalg.norm(node.position - self.goal_pos) <= self.goal_tolerance
        return position_close
    
    def is_goal_reached_with_stop(self, node):
        """
        Check if the goal is reached with near-zero velocity (complete stop).
        
        Args:
            node (KNode): Node to check
            
        Returns:
            bool: True if goal is reached with low velocity
        """
        position_close = np.linalg.norm(node.position - self.goal_pos) <= self.goal_tolerance
        velocity_low = np.linalg.norm(node.velocity) <= 0.5  # Low velocity threshold
        return position_close and velocity_low
    
    def plan(self):
        """
        Execute the Kinodynamic RRT planning algorithm.
        
        Returns:
            list or None: Path from start to goal as list of nodes, None if no path found
        """
        print("Starting kinodynamic RRT algorithm")
        
        for iteration in range(self.max_iterations): #max iterations given by user 
            if iteration % 100 == 0: #print every 100 iterations 
                print(f"Iteration {iteration}/{self.max_iterations}")
            
            # Every 10th iteration, try to connect closest node to goal directly
            if iteration % 10 == 0 and iteration > 0:
                # Find closest node by position only (any velocity)
                closest_node = self.find_nearest_node_by_position(self.goal_pos)
                
                # Try to connect this node to goal
                goal_node = self.try_connect_to_goal(closest_node)
                
                if goal_node is not None:
                    # Successfully connected to goal!
                    self.nodes.append(goal_node)
                    self.goal_node = goal_node
                    print(f"Goal reached directly in {iteration + 1} iterations!")
                    print(f"Connected from node at {closest_node.position} with velocity {closest_node.velocity}")
                    print(f"Final velocity at goal: {goal_node.velocity}")
                    return self.extract_path()
            
            # Sample random state
            rand_pos, rand_vel = self.sample_random_state()
            
            # Find nearest node (considering velocity)
            nearest_node = self.find_nearest_node(rand_pos, rand_vel)
            
            # Steer towards random position
            new_node = self.steer(nearest_node, rand_pos)
            
            if new_node is None:
                continue
            
            # Check collision
            if not self.check_collision(nearest_node.position, new_node.position):
                continue
            
            # Add node to tree
            self.nodes.append(new_node)
            
            # Check if goal is reached
            if self.is_goal_reached(new_node):
                # Try to connect directly to the goal with stopping
                goal_node = self.steer_to_goal_with_stop(new_node, self.goal_pos)
                
                if goal_node is not None and self.check_collision(new_node.position, goal_node.position):
                    # Successfully connected to goal
                    goal_node.position = self.goal_pos  # Ensure exact goal position
                    # Keep the velocity and acceleration as calculated by steer_to_goal_with_stop
                    self.nodes.append(goal_node)
                    self.goal_node = goal_node
                    print(f"Goal reached in {iteration + 1} iterations!")
                    return self.extract_path()
                else:
                    # Can't connect directly, but close enough
                    self.goal_node = new_node
                    print(f"Goal reached in {iteration + 1} iterations (within tolerance)!")
                    return self.extract_path()
        
        print("Maximum iterations reached. No path found.")
        return None
    
    def extract_path(self):
        """
        Extract path from start to goal.
        
        Returns:
            list: List of nodes from start to goal
        """
        if self.goal_node is None:
            return None
        
        path = []
        current = self.goal_node
        
        while current is not None: #just run backwards from goal to start
            path.append(current)
            current = current.parent
        
        path.reverse()
        
        # Don't modify the final node - it should have the velocity/acceleration
        # that resulted from the physics-based connection
        
        return path
    


