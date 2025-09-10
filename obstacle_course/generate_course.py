import numpy as np

class ObstacleCourse:
    def __init__(self, n:int , m: int, num_obstacles: int, min_dim: int = 10, obstacle_size: int = 2):
        """
        Initialize the obstacle course.
        
        Args:
            n (int): Width of the course
            m (int): Height of the course
            num_obstacles (int): Number of obstacles to generate
            min_dim (int): Minimum allowed dimension size
            obstacle_size (int): Size of square obstacles (c x c)
        """
        if n < min_dim or m < min_dim:
            raise ValueError(f"Course dimensions must be at least {min_dim}x{min_dim}")
        
        if obstacle_size >= min(n, m):
            raise ValueError(f"Obstacle size {obstacle_size} is too large for course dimensions {n}x{m}")
        
        self.width = n
        self.height = m
        self.obstacle_size = obstacle_size
        self.num_obstacles = num_obstacles
        self.grid = np.zeros((m, n), dtype=int)  # 0 represents free space, 1 represents obstacle
        
        # Initialize start and goal positions
        self.start = None
        self.goal = None
        
    def generate_obstacles(self):
        """Generate random obstacles in the course."""
        obstacles_placed = 0
        max_attempts = 1e3  # Prevent infinite loop
        attempts = 0
        
        while obstacles_placed < self.num_obstacles and attempts < max_attempts:
            # Generate random position for obstacle's top-left corner
            x = np.random.randint(0, self.width - self.obstacle_size + 1)
            y = np.random.randint(0, self.height - self.obstacle_size + 1)
            
            # Check if space is free
            if not self._check_overlap(x, y):
                # Place obstacle
                self.grid[y:y+self.obstacle_size, x:x+self.obstacle_size] = 1
                obstacles_placed += 1
            
            attempts += 1
            
        if obstacles_placed < self.num_obstacles:
            print(f"Warning: Could only place {obstacles_placed} obstacles out of {self.num_obstacles} requested")
    
    def _check_overlap(self, x: int, y: int) -> bool:
        """
        Check if placing an obstacle at (x,y) would overlap with existing obstacles.
        
        Returns:
            bool: True if there's an overlap, False otherwise
        """
        return np.any(self.grid[y:y+self.obstacle_size, x:x+self.obstacle_size] == 1)
    
    def get_grid(self) -> np.ndarray:
        """Return the course grid."""
        return self.grid.copy()
    
    def generate_start_goal(self, margin: int = 2):
        """
        Generate start and goal points.
        Start point will be in the bottom-left corner region.
        Goal point will be in the top-right corner region.
        
        Args:
            margin (int): Minimum distance from the edges
        """
        # Define regions for start and goal
        start_x_range = (margin, self.width // 4)  # Left quarter of the grid
        start_y_range = (self.height * 3 // 4, self.height - margin)  # Bottom quarter
        
        goal_x_range = (self.width * 3 // 4, self.width - margin)  # Right quarter
        goal_y_range = (margin, self.height // 4)  # Top quarter
        
        # Generate start point in bottom-left region
        while True:
            start_x = np.random.randint(start_x_range[0], start_x_range[1])
            start_y = np.random.randint(start_y_range[0], start_y_range[1])
            if not self._check_point_collision(start_x, start_y):
                self.start = (start_x, start_y)
                break
        
        # Generate goal point in top-right region
        while True:
            goal_x = np.random.randint(goal_x_range[0], goal_x_range[1])
            goal_y = np.random.randint(goal_y_range[0], goal_y_range[1])
            if not self._check_point_collision(goal_x, goal_y):
                self.goal = (goal_x, goal_y)
                break
    
    def _check_point_collision(self, x: int, y: int) -> bool:
        """
        Check if a point collides with any obstacle.
        
        Args:
            x (int): x-coordinate
            y (int): y-coordinate
            
        Returns:
            bool: True if point collides with an obstacle, False otherwise
        """
        return self.grid[y, x] == 1
    
    def get_start_goal(self) -> tuple:
        """
        Get the start and goal positions.
        
        Returns:
            tuple: ((start_x, start_y), (goal_x, goal_y))
        """
        if self.start is None or self.goal is None:
            self.generate_start_goal()
        return self.start, self.goal
