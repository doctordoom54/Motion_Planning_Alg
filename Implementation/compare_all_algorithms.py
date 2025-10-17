import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure parent directory is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course.generate_course import ObstacleCourse
from obstacle_course.course_visualizer import CourseVisualizer
from KD_RRT import KinodynamicRRT
from rrt_star import RRTStar
from MCTS import MCTSPlanner


def compute_path_length(points):
    if points is None or len(points) < 2:
        return math.inf
    length = 0.0
    for i in range(1, len(points)):
        p0 = np.array(points[i-1])
        p1 = np.array(points[i])
        length += float(np.linalg.norm(p1 - p0))
    return length


def run_kd_rrt(course, start_pos, goal_pos):
    start_vel = [0, 0.0]
    start_acc = [0, 0]
    kd_planner = KinodynamicRRT(
        start_pos=start_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        goal_pos=goal_pos,
        course=course,
        max_iterations=10000,
        step_size=1.5,
        goal_tolerance=1
    )
    t0 = time.time()
    path = kd_planner.plan()
    runtime = time.time() - t0
    cost = path[-1].cost if path else None
    # Extract simple point list for plotting/metrics
    pts = [node.position for node in path] if path else None
    return {
        'name': 'Kinodynamic RRT',
        'path_nodes': path,
        'points': pts,
        'cost': cost,
        'runtime': runtime,
        'extra': {
            'tree_size': len(kd_planner.nodes)
        },
        'planner': kd_planner
    }


def run_rrt_star(course, start_pos, goal_pos):
    map_bounds = (0, course.width, 0, course.height)
    rrt_star = RRTStar(
        start=start_pos,
        goal=goal_pos,
        obstacle_list=course.obstacles,
        map_bounds=map_bounds,
        step_size=4.0,
        goal_sample_rate=0.5,
        max_iter=5000
    )
    t0 = time.time()
    path = rrt_star.plan()
    runtime = time.time() - t0
    cost = compute_path_length(path) if path else None
    return {
        'name': 'RRT*',
        'path_nodes': None,  # RRT* returns list of points
        'points': path,
        'cost': cost,
        'runtime': runtime,
        'extra': {
            'tree_size': len(rrt_star.node_list)
        },
        'planner': rrt_star
    }


def run_mcts(course, start_pos, goal_pos):
    start_vel = (0, 0.00)
    start_acc = (0.0, 0.0)
    mcts = MCTSPlanner(
         course=course,
        start_pos=start_pos,
        goal_pos=goal_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        dt=1,  # Reasonable time step
        vmax=35,  # Much lower max velocity for better control
        amax=5,  # Higher acceleration for movement
        amin=-5,  # Symmetric acceleration bounds
        goal_tolerance=1, 
        goal_vel_tolerance= 0.5, # Larger tolerance for easier goal reaching
        uct_c=np.sqrt(2),  
        widen_k=2,  # Allow more children per node
        widen_alpha=0.2,  # Slower growth = more exploration early
        rollout_horizon=20,  # Shorter horizon to avoid getting stuck
        max_iterations=int(4*1e4),  # Reduce for quick testing
        direct_connect_radius=5.0,  
        
    )
    # Adjust reward scales (matching existing usage)
    mcts.reward_goal = 100.0
    mcts.reward_collision = -100.0
    mcts.reward_progress = 50

    t0 = time.time()
    path = mcts.plan()
    runtime = time.time() - t0
    pts = path if path else None  # MCTS returns list of (x,y)
    cost = compute_path_length(pts) if pts else None
    return {
        'name': 'MCTS',
        'path_nodes': None,
        'points': pts,
        'cost': cost,
        'runtime': runtime,
        'extra': {
            'iterations': mcts.max_iterations
        },
        'planner': mcts
    }


def summarize_result(res, goal_pos):
    pts = res['points']
    if pts and len(pts) > 0:
        end = np.array(pts[-1])
        dist_goal = float(np.linalg.norm(end - np.array(goal_pos)))
    else:
        dist_goal = math.inf
    reached = dist_goal <= 5.0  # generic threshold for summary
    return {
        'Algorithm': res['name'],
        'Nodes': len(pts) if pts else 0,
        'Cost': res['cost'] if res['cost'] is not None else math.inf,
        'Runtime_s': res['runtime'],
        'End_Dist_to_Goal': dist_goal,
        'Reached?': reached,
        **res['extra']
    }


def print_summary_table(summaries):
    # Determine columns dynamically
    columns = []
    for s in summaries:
        for k in s.keys():
            if k not in columns:
                columns.append(k)
    # Fixed order preference
    pref = ['Algorithm', 'Nodes', 'Cost', 'Runtime_s', 'End_Dist_to_Goal', 'Reached?', 'tree_size', 'iterations']
    columns = [c for c in pref if c in columns] + [c for c in columns if c not in pref]
    # Print header
    header = ' | '.join(f"{c:>15}" for c in columns)
    print('\nSUMMARY:')
    print(header)
    print('-' * len(header))
    for s in summaries:
        row = ' | '.join(f"{(s.get(c, '')):>15}" for c in columns)
        print(row)


def visualize(course, start_pos, goal_pos, results):
    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    titles = [r['name'] for r in results]
    for ax, res, title in zip(axes, results, titles):
        viz = CourseVisualizer(figsize=(9, 9))
        viz.fig = fig
        viz.ax = ax
        viz.plot_obstacles(
            obstacles=course.obstacles,
            width=course.width,
            height=course.height,
            start=start_pos,
            goal=goal_pos,
            title=title
        )
        pts = res['points']
        if pts and len(pts) > 1:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, 'r-', linewidth=2, label=f"{title} Path")
            ax.legend()
        # Optionally draw trees (KD-RRT & RRT*)
        if res['name'] == 'Kinodynamic RRT':
            planner = res['planner']
            for node in planner.nodes[1:]:
                if node.parent is not None:
                    ax.plot([node.parent.position[0], node.position[0]],
                            [node.parent.position[1], node.position[1]],
                            'b-', alpha=0.15, linewidth=0.5)
        if res['name'] == 'RRT*':
            planner = res['planner']
            for node in planner.node_list:
                if node.parent is not None:
                    ax.plot([node.parent.position[0], node.position[0]],
                            [node.parent.position[1], node.position[1]],
                            'b-', alpha=0.15, linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main():
    # Single random course for all algorithms
    course = ObstacleCourse(100, 100, 10, obstacle_size=10)
    course.generate_obstacles()
    course.set_start_and_goal()

    start_pos = course.start
    goal_pos = course.goal
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print('=' * 60)

    # Run algorithms
    results = []
    print('[1/3] Running Kinodynamic RRT...')
    results.append(run_kd_rrt(course, start_pos, goal_pos))
    print('[2/3] Running RRT*...')
    results.append(run_rrt_star(course, start_pos, goal_pos))
    print('[3/3] Running MCTS...')
    results.append(run_mcts(course, start_pos, goal_pos))

    # Summaries
    summaries = [summarize_result(r, goal_pos) for r in results]
    print_summary_table(summaries)

    # Visualization
    try:
        visualize(course, start_pos, goal_pos, results)
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == '__main__':
    main()
