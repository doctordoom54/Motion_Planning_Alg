import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from KD_RRT import KinodynamicRRT
from rrt_star import RRTStar, Node


def main():
    """
    Compare Kinodynamic RRT and RRT* algorithms on the same obstacle course.
    """
    # Create obstacle course (same for both algorithms)
    course = ObstacleCourse(200, 200, 20, obstacle_size=18)
    course.generate_obstacles()
    course.set_start_and_goal()
    
    # Get start and goal from course
    start_pos = course.start
    goal_pos = course.goal
    
    # Set initial velocity and acceleration for kinodynamic RRT
    start_vel = [-0.35, -1.0]
    start_acc = [1.0, -1.0]
    
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print("=" * 50)
    
    # =================== KINODYNAMIC RRT ===================
    print("Running Kinodynamic RRT...")
    kd_planner = KinodynamicRRT(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        goal_pos=goal_pos,
        course=course,
        max_iterations=10000,
        step_size=0.6,
        goal_tolerance=5
    )
    
    kd_path = kd_planner.plan()
    
    # Print KD-RRT results
    if kd_path is not None:
        kd_cost = kd_path[-1].cost
        print(f"KD-RRT: Path found with {len(kd_path)} nodes, Total cost: {kd_cost:.2f}")
    else:
        print("KD-RRT: No path found")
        kd_cost = None
    
    # =================== RRT* ===================
    print("Running RRT*...")
    map_bounds = (0, course.width, 0, course.height)
    rrt_star = RRTStar(
        start=start_pos,  # Use same start position
        goal=goal_pos,    # Use same goal position
        obstacle_list=course.obstacles,
        map_bounds=map_bounds,
        step_size=4.0,
        goal_sample_rate=0.5,
        max_iter=5000
    )
    
    rrt_path = rrt_star.plan()
    
    # Print RRT* results
    if rrt_path is not None:
        # Calculate RRT* path cost
        rrt_cost = 0
        for i in range(1, len(rrt_path)):
            rrt_cost += np.linalg.norm(np.array(rrt_path[i]) - np.array(rrt_path[i-1]))
        print(f"RRT*: Path found with {len(rrt_path)} nodes, Total cost: {rrt_cost:.2f}")
    else:
        print("RRT*: No path found")
        rrt_cost = None
    
    # =================== SIDE-BY-SIDE VISUALIZATION ===================
    # Create separate CourseVisualizers for side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Kinodynamic RRT (Left subplot)
    visualizer1 = CourseVisualizer(figsize=(10, 10))
    visualizer1.fig = fig  # Use our subplot figure
    visualizer1.ax = ax1   # Use our left subplot
    
    visualizer1.plot_obstacles(
        obstacles=course.obstacles,
        width=course.width,
        height=course.height,
        start=start_pos,
        goal=goal_pos,
        title="Kinodynamic RRT"
    )
    
    # Plot KD-RRT tree
    for node in kd_planner.nodes[1:]:
        if node.parent is not None:
            x_coords = [node.parent.position[0], node.position[0]]
            y_coords = [node.parent.position[1], node.position[1]]
            ax1.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot KD-RRT path
    if kd_path is not None:
        path_x = [node.position[0] for node in kd_path]
        path_y = [node.position[1] for node in kd_path]
        ax1.plot(path_x, path_y, 'r-', linewidth=3, label=f'KD-RRT Path')
        ax1.legend()
    
    # Plot RRT* (Right subplot)
    visualizer2 = CourseVisualizer(figsize=(10, 10))
    visualizer2.fig = fig  # Use our subplot figure
    visualizer2.ax = ax2   # Use our right subplot
    
    visualizer2.plot_obstacles(
        obstacles=course.obstacles,
        width=course.width,
        height=course.height,
        start=start_pos,
        goal=goal_pos,
        title="RRT*"
    )
    
    # Plot RRT* tree
    for node in rrt_star.node_list:
        if node.parent is not None:
            x_vals = [node.position[0], node.parent.position[0]]
            y_vals = [node.position[1], node.parent.position[1]]
            ax2.plot(x_vals, y_vals, 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot RRT* path
    if rrt_path is not None:
        px, py = zip(*rrt_path)
        ax2.plot(px, py, 'r-', linewidth=3, label=f'RRT* Path')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()