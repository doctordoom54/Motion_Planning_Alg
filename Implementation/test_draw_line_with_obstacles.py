import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to Python path so we can import from obstacle_course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer

# Parameters
width = 50
height = 50
num_obstacles = 8
obstacle_size = 4

# Generate course
course = ObstacleCourse(n=width, m=height, num_obstacles=num_obstacles, obstacle_size=obstacle_size)
course.generate_obstacles()
course.set_start_and_goal()

# Sample a random free point (not in obstacle)
rand_point = course.sample_free_point()

# Visualize
visualizer = CourseVisualizer(figsize=(12, 12))
visualizer.plot_obstacles(
    obstacles=course.obstacles,
    width=course.width,
    height=course.height,
    start=course.start,
    goal=course.goal,
    title="Line from Start to Random Free Point"
)
# Draw the line from start to random point
visualizer.ax.plot([course.start[0]+0.5, rand_point[0]+0.5], [course.start[1]+0.5, rand_point[1]+0.5], 'b-', label='Line')
visualizer.ax.legend()
visualizer.show()
