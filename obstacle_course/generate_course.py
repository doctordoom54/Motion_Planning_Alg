import numpy as np

class ObstacleCourse:
    def __init__(self, n: int, m: int, num_obstacles: int, min_dim: int = 10, obstacle_size: int = 2):
        """
        Initialize a continuous obstacle course with c x c square obstacles.
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
        self.obstacles = []  # List of (x, y, c) tuples, (bottom-left corner, size)
        self.start = None
        self.goal = None

    def generate_obstacles(self):
        """Generate random c x c obstacles as (x, y, c) tuples (bottom-left corner, size)."""
        obstacles_placed = 0
        max_attempts = 1000
        attempts = 0
        while obstacles_placed < self.num_obstacles and attempts < max_attempts:
            x = np.random.randint(0, self.width - self.obstacle_size + 1)
            y = np.random.randint(0, self.height - self.obstacle_size + 1)
            if not self._check_overlap(x, y):
                self.obstacles.append((x, y, self.obstacle_size))
                obstacles_placed += 1
            attempts += 1
        if obstacles_placed < self.num_obstacles:
            print(f"Warning: Could only place {obstacles_placed} obstacles out of {self.num_obstacles} requested")

    def _check_overlap(self, x: int, y: int) -> bool:
        """Check if a new c x c block at (x, y) would overlap with any existing obstacle."""
        c = self.obstacle_size
        for ox, oy, oc in self.obstacles:
            # Check for rectangle overlap
            if (x < ox + oc and x + c > ox and y < oy + oc and y + c > oy):
                return True
        return False

    def sample_free_point(self):
        """Sample a random integer point not inside any obstacle."""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if not self.point_in_obstacle(x, y):
                return (x, y)
        raise RuntimeError("Could not sample a free point after many attempts.")

    def point_in_obstacle(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside any obstacle."""
        for ox, oy, c in self.obstacles:
            if ox <= x < ox + c and oy <= y < oy + c:
                return True
        return False

    def set_start_and_goal(self, corner_box_size: int = 8):
        """
        Set start and goal points randomly within a small region near the corners, not exactly at the corners, and not inside any obstacle.
        corner_box_size (int): Size of the square region from each corner to sample start/goal.
        """
        # Define corner regions
        regions = [
            (0, 0, corner_box_size, corner_box_size),  # bottom-left
            (self.width - corner_box_size, 0, self.width, corner_box_size),  # bottom-right
            (0, self.height - corner_box_size, corner_box_size, self.height),  # top-left
            (self.width - corner_box_size, self.height - corner_box_size, self.width, self.height)  # top-right
        ]
        # Sample start in one region, goal in another (not the same)
        max_attempts = 1000
        for _ in range(max_attempts):
            start_region = regions[0]
            goal_region = regions[-1]
            sx = np.random.randint(start_region[0], start_region[2])
            sy = np.random.randint(start_region[1], start_region[3])
            gx = np.random.randint(goal_region[0], goal_region[2])
            gy = np.random.randint(goal_region[1], goal_region[3])
            if not self.point_in_obstacle(sx, sy) and not self.point_in_obstacle(gx, gy) and (sx, sy) != (gx, gy):
                self.start = (sx, sy)
                self.goal = (gx, gy)
                return
        raise RuntimeError("Could not find free start and goal points near corners.")
    
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
