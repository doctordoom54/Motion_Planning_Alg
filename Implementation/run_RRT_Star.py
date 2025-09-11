#viz(250, 250, 20, 15)
print("Starting run_RRT_Star.py")
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from rrt_star import RRTStar, Node

print("Imported RRT_Star module")
def main():
    print("Entered main()")
    # Parameters
    width = 250
    height = 500
    num_obstacles = 35
    obstacle_size = 25
    step_size = 10.0
    max_iter = 3000

    # Generate course
    course = ObstacleCourse(n=width, m=height, num_obstacles=num_obstacles, obstacle_size=obstacle_size)
    course.generate_obstacles()
    course.set_start_and_goal()

    # RRT* planning
    map_bounds = (0, width, 0, height)
    
    rrt = RRTStar(
        start=course.start,
        goal=course.goal,
        obstacle_list=course.obstacles,
        map_bounds=map_bounds,
        step_size=step_size,
        goal_sample_rate=0.5,
        max_iter=max_iter
    )
    print("About to call rrt.plan()")
    path = rrt.plan()

    # Visualization
    visualizer = CourseVisualizer(figsize=(10, 10))
    visualizer.plot_obstacles(
        obstacles=course.obstacles,
        width=width,
        height=height,
        start=course.start,
        goal=course.goal,
        title="RRT* Path Planning"
    )
    # Draw the whole tree (all edges)
    for node in rrt.node_list:
        if node.parent is not None:
            x_vals = [node.position[0], node.parent.position[0]]
            y_vals = [node.position[1], node.parent.position[1]]
            visualizer.ax.plot(x_vals, y_vals, color='gray', linewidth=0.7, alpha=0.7)

    # Draw the path if found
    if path and len(path) > 1:
        px, py = zip(*path)
        visualizer.ax.plot(px, py, 'b-', linewidth=2, label='RRT* Path')
        visualizer.ax.legend()
    visualizer.show()

if __name__ == "__main__":
    main()
