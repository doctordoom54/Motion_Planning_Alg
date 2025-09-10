import sys
import os

# Add the parent directory to Python path so we can import from Obstacle Course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course import ObstacleCourse, CourseVisualizer



def viz(width, height, num_obstacles, obstacle_size):
    # Create and generate the obstacle course
    course = ObstacleCourse(
        n=width,
        m=height,
        num_obstacles=num_obstacles,
        obstacle_size=obstacle_size
    )
    course.generate_obstacles()
    course.set_start_and_goal()
    # Create visualizer and display the course
    visualizer = CourseVisualizer(figsize=(12, 12))
    visualizer.plot_obstacles(
        obstacles=course.obstacles,
        width=course.width,
        height=course.height,
        start=course.start,
        goal=course.goal,
        title="Obstacle Course"
    )
    visualizer.show()

if __name__ == "__main__":
    viz()
