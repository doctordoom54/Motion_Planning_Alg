import sys
import os

# Add the parent directory to Python path so we can import from Obstacle Course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course import ObstacleCourse, CourseVisualizer


def viz(width, height, num_obstacles, obstacle_size):
    # Define course parameters
    # Create and generate the obstacle course
    course = ObstacleCourse(
        n=width, 
        m=height, 
        num_obstacles=num_obstacles,
        obstacle_size=obstacle_size
    )
    
    # Generate the obstacles
    course.generate_obstacles()
    start, goal = course.get_start_goal()
    
    # Create visualizer and display the course
    grid = CourseVisualizer(figsize=(12, 12))  # Larger figure size for better visibility
    grid.plot_grid(course.get_grid(), start= start, goal= goal, title="Obstacle Course")
    return grid.show()

    #viz.close()

if __name__ == "__main__":
    viz()
