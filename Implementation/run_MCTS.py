print("Starting run_MCTS.py")
import sys
import os
import numpy as np

# Add the parent directory to Python path so we can import from obstacle_course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from MCTS import MCTSPlanner


def main():
    # Parameters for the obstacle course
    width = 50
    height = 50
    num_obstacles = 7
    obstacle_size = 5

    # Create obstacle course
    course = ObstacleCourse(n=width, m=height, num_obstacles=num_obstacles, obstacle_size=obstacle_size)
    course.generate_obstacles()
    course.set_start_and_goal()

    print(f"Start: {course.start}, Goal: {course.goal}")
    print(f"Distance from boundaries: Start=({course.start[0]}, {course.start[1]}), Goal=({course.goal[0]}, {course.goal[1]})")

    # Optional: non-zero initial velocity/acceleration
    start_vel = (0.0, 0.0)
    start_acc = (0.0, 0.0)

    # Build MCTS planner with parameters suitable for large distances
    planner = MCTSPlanner(
        course=course,
        start_pos=course.start,
        goal_pos=course.goal,
        start_vel=start_vel,
        start_acc=start_acc,
        dt=1,  # Reasonable time step
        vmax=4.50,  # Much lower max velocity for better control
        amax=8,  # Higher acceleration for movement
        amin=0,  # Symmetric acceleration bounds
        goal_tolerance=3.0,  # Larger tolerance for easier goal reaching
        uct_c=1.2,
        widen_k=5.0,  # Allow more children per node
        widen_alpha=0.2,  # Slower growth = more exploration early
        rollout_horizon=20,  # Shorter horizon to avoid getting stuck
        max_iterations=int(7*1e5),  # Reduce for quick testing
        direct_connect_radius=5.0,  
        goal_bias_expand=0.8,
        goal_bias_rollout=0.9,
        goal_accel_scale=1.0,
        
    )

    # Adjust reward scales to not overly punish early collisions
    planner.reward_goal = 100.0
    planner.reward_collision = -100.0
    planner.reward_step = 50

    # Run planning
    print("Running MCTS planner...")
    try:
        path = planner.plan()
        
        # Check if we actually found a goal path
        if path and len(path) > 1:
            distance_to_goal = np.linalg.norm(np.array(path[-1]) - np.array(course.goal))
            if distance_to_goal <= planner.goal_tolerance:
                print("Planning completed - GOAL REACHED!")
            else:
                print(f"Planning completed - but goal NOT reached (distance: {distance_to_goal:.3f})")
        else:
            print("✓ Planning completed - but NO PATH found")
            
    except Exception as e:
        print(f"Planning failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print results
    if path and len(path) > 1:
        print(f"Path length (nodes): {len(path)}")
        # Use the same tolerance as the planner for consistency
        distance_to_goal = np.linalg.norm(np.array(path[-1]) - np.array(course.goal))
        is_goal = distance_to_goal <= planner.goal_tolerance
        print(f"Ends at goal: {is_goal}; last point: {path[-1]}; goal: {course.goal}")
        print(f"Distance to goal: {distance_to_goal:.3f} (tolerance: {planner.goal_tolerance})")
    else:
        print("No path found by MCTS.")
        return

    # Visualization (with error handling)
    try:
        print("Creating visualization...")
        visualizer = CourseVisualizer(figsize=(10, 10))
        visualizer.plot_obstacles(
            obstacles=course.obstacles,
            width=course.width,
            height=course.height,
            start=course.start,
            goal=course.goal,
            title="MCTS Path Planning"
        )

        if path and len(path) > 1:
            px, py = zip(*path)
            visualizer.ax.plot(px, py, 'r-', linewidth=2.5, label='MCTS Path')
            visualizer.ax.legend()

        print("Showing plot...")
        visualizer.show()
        print("✓ Visualization completed")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        print("Planning was successful, but could not display the plot.")


if __name__ == "__main__":
    main()
