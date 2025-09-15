print("Starting run_KD_RRT.py")
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from KD_RRT import KinodynamicRRT



def main():
    """
    Example usage of the Kinodynamic RRT planner.
    """
    # Create obstacle course
    course = ObstacleCourse(100,200, 18, obstacle_size=15)
    course.generate_obstacles()
    course.set_start_and_goal()
    
    # Get start and goal from course
    start_pos = course.start
    goal_pos = course.goal
    
    # Set initial velocity and acceleration
    start_vel = [-0.35, -1.0]  # Start from rest
    start_acc = [1.0, 0.0]  # No initial acceleration
    
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    
    # Create and run planner
    planner = KinodynamicRRT(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        goal_pos=goal_pos,
        course=course,
        max_iterations=10000,
        step_size=0.6,
        goal_tolerance=5
    )
    
    # output planned path 
    path = planner.plan()
    
    if path:
        print(f"Path found with {len(path)} nodes")
        print("Path details:")
        for i, node in enumerate(path):
            pos_str = f"[{node.position[0]:.2f}, {node.position[1]:.2f}]"
            vel_str = f"[{node.velocity[0]:.2f}, {node.velocity[1]:.2f}]"
            acc_str = f"[{node.acceleration[0]:.2f}, {node.acceleration[1]:.2f}]"
            print(f"  Node {i}: pos={pos_str}, vel={vel_str}, acc={acc_str}")
    else:
        print("No path found")
    
    # Simple visualization - just path
    visualizer = CourseVisualizer(figsize=(10, 10))
    visualizer.plot_obstacles(
        course.obstacles,
        course.width,
        course.height,
        course.start,
        course.goal,
        "Kinodynamic RRT Path"
    )
    
    # Plot tree edges
    for node in planner.nodes[1:]:  # Skip root
        if node.parent is not None:
            x_coords = [node.parent.position[0], node.position[0]]
            y_coords = [node.parent.position[1], node.position[1]]
            plt.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot nodes
    for node in planner.nodes:
        plt.plot(node.position[0], node.position[1], 'b.', markersize=2)
    
    # Plot path if provided
    if path is not None:
        path_x = [node.position[0] for node in path]
        path_y = [node.position[1] for node in path]
        plt.plot(path_x, path_y, 'r-', linewidth=3, label='Path')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.title('Kinodynamic RRT Planning')
    plt.show()

if __name__ == "__main__":
    main()