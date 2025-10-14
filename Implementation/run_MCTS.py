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
    width = 80
    height = 80
    num_obstacles = 6
    obstacle_size = 10

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
        dt=0.5,  # Reasonable time step
        vmax=35,  # Much lower max velocity for better control
        amax=5,  # Higher acceleration for movement
        amin=-5,  # Symmetric acceleration bounds
        goal_tolerance=1,  # Larger tolerance for easier goal reaching
        uct_c=np.sqrt(2),  
        widen_k=6,  # Allow more children per node
        widen_alpha=0.2,  # Slower growth = more exploration early
        rollout_horizon=25,  # Shorter horizon to avoid getting stuck
        max_iterations=int(1e5),  # Reduce for quick testing
        direct_connect_radius=5.0,  
         #basic heuristic to accelerate towards goal
        
        
        
    )

    # Adjust reward scales to not overly punish early collisions
    planner.reward_goal = 100.0
    planner.reward_collision = -80.0
    planner.reward_progress = 70

    # Run planning
    print("Running MCTS planner...")
    try:
        path = planner.plan()
        
        # Store path in planner for later access
        #planner.final_path = path
        
        # Extract full trace with position, velocity, acceleration, and actions
        full_trace = planner.extract_full_trace()
        #planner.full_trace = full_trace
        
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
        return None, None, None

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
        return None, None, None

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

        # Plot all explored paths (MCTS tree) with thin black lines
        def plot_tree(node):
            """Recursively plot all edges in the MCTS tree"""
            for child in node.children:
                # Draw edge from parent to child
                visualizer.ax.plot(
                    [node.position[0], child.position[0]], 
                    [node.position[1], child.position[1]], 
                    'k-', linewidth=0.3, alpha=0.4
                )
                # Recursively plot children
                plot_tree(child)
        
        # Plot the entire tree starting from root
        print("Plotting MCTS tree exploration...")
        plot_tree(planner.root)

        # Plot the best path on top in red
        if path and len(path) > 1:
            px, py = zip(*path)
            visualizer.ax.plot(px, py, 'r-', linewidth=2, label='Best Path', zorder=10)
            visualizer.ax.legend(loc='upper right')

        print("Showing plot...")
        visualizer.show()
        print("✓ Visualization completed")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        print("Planning was successful, but could not display the plot.")
    
    # Return the planner, course, and path for later use
    return planner, course, path, full_trace


if __name__ == "__main__":
    # Run main and optionally capture results
    result = main()
    
    # If main() returned values, unpack them for interactive use
    if result is not None:
        planner, course, path , full_trace = result
        print("\n" + "="*60)
        print("Results stored in variables for later use:")
        print("  - planner: MCTSPlanner instance")
        print("  - course: ObstacleCourse instance")
        print("  - path: List of (x, y) position tuples")
        print("  - planner.final_path: Same as path")
        print("  - planner.full_trace: List of dicts with full state info")
        print("\nAccess full trace info:")
        print("  planner.full_trace[i]['position']     - position at step i")
        print("  planner.full_trace[i]['velocity']     - velocity at step i")
        print("  planner.full_trace[i]['acceleration'] - acceleration at step i")
        print("  planner.full_trace[i]['action']       - action applied at step i")
        print("  - path: List of (x, y) tuples")
        print("  - planner.final_path: Same as path")
        print("="*60)
