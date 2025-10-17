import os
import sys
import time
import math
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course.generate_course import ObstacleCourse
from KD_RRT import KinodynamicRRT
from MCTS import MCTSPlanner


def _run_kd_rrt(course, start_pos, goal_pos):
    """
    Run Kinodynamic RRT on the given course.
    Returns dict with success flag and total action.
    """
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
    
    if path is not None:
        # Calculate total action (sum of acceleration norms)
        total_action = 0.0
        for node in path:
            acc_norm = np.linalg.norm(node.acceleration)
            total_action += acc_norm
        
        return {
            'converged': 1,
            'total_action': total_action,
            'runtime': runtime,
            'path_length': len(path),
            'cost': path[-1].cost
        }
    else:
        return {
            'converged': 0,
            'total_action': np.nan,
            'runtime': runtime,
            'path_length': 0,
            'cost': np.nan
        }


def _run_mcts(course, start_pos, goal_pos):
    """
    Run MCTS on the given course.
    Returns dict with success flag and total action.
    """
    start_vel = (0, 0.00)
    start_acc = (0.0, 0.0)
    mcts = MCTSPlanner(
        course=course,
        start_pos=start_pos,
        goal_pos=goal_pos,
        start_vel=start_vel,
        start_acc=start_acc,
        dt=1,
        vmax=35,
        amax=5,
        amin=-5,
        goal_tolerance=1,
        goal_vel_tolerance=0.5,
        uct_c=np.sqrt(2),
        widen_k=2,
        widen_alpha=0.2,
        rollout_horizon=20,
        max_iterations=int(4*1e4),
        direct_connect_radius=5.0,
    )
    
    # Reward scales
    mcts.reward_goal = 100.0
    mcts.reward_collision = -100.0
    mcts.reward_progress = 50

    t0 = time.time()
    path = mcts.plan()
    runtime = time.time() - t0
    
    if path is not None and len(path) > 0:
        # Calculate total action from MCTS best path using actions (not accelerations)
        # Match the calculation in interactive_mcts_explorer.py
        total_action = 0.0
        if hasattr(mcts, 'root') and mcts.root is not None:
            # Find best path through tree (node with highest visits at each level)
            best_path = []
            current = mcts.root
            while current is not None:
                best_path.append(current)
                if hasattr(current, 'children') and len(current.children) > 0:
                    # Pick child with highest visits
                    current = max(current.children, key=lambda c: c.visits if hasattr(c, 'visits') else 0)
                    if current.visits == 0:
                        break
                else:
                    break
            
            # Calculate total action as sum of action norms (skip root which has no action)
            for i, node in enumerate(best_path):
                if i > 0 and hasattr(node, 'action') and node.action is not None:
                    action_norm = np.linalg.norm(node.action)
                    total_action += action_norm
        
        return {
            'converged': 1,
            'total_action': total_action,
            'runtime': runtime,
            'path_length': len(path),
            'cost': np.nan  # MCTS doesn't track cost
        }
    else:
        return {
            'converged': 0,
            'total_action': np.nan,
            'runtime': runtime,
            'path_length': 0,
            'cost': np.nan
        }


def compare_algorithms_monte_carlo(num_trials, course_width=100, course_height=100, 
                                    num_obstacles=10, obstacle_size=10):
    """
    Run Monte Carlo comparison between KD-RRT and MCTS.
    
    Args:
        num_trials: Number of random courses to test
        course_width: Width of the course (default: 100)
        course_height: Height of the course (default: 100)
        num_obstacles: Number of obstacles (default: 10)
        obstacle_size: Size of obstacles (default: 10)
    
    Returns:
        pandas.DataFrame with columns:
            - trial: Trial number
            - kd_rrt_converged: 1 if KD-RRT found path, 0 otherwise
            - kd_rrt_total_action: Sum of acceleration norms for KD-RRT path
            - kd_rrt_runtime: Runtime in seconds for KD-RRT
            - kd_rrt_path_length: Number of nodes in KD-RRT path
            - kd_rrt_cost: Path cost for KD-RRT
            - mcts_converged: 1 if MCTS found path, 0 otherwise
            - mcts_total_action: Sum of acceleration norms for MCTS path
            - mcts_runtime: Runtime in seconds for MCTS
            - mcts_path_length: Number of nodes in MCTS path
    """
    results = []
    
    for trial in range(num_trials):
        # Create a new random course for this trial
        course = ObstacleCourse(course_width, course_height, num_obstacles, obstacle_size=obstacle_size)
        course.generate_obstacles()
        course.set_start_and_goal()
        
        start_pos = course.start
        goal_pos = course.goal
        
        # Run KD-RRT
        kd_result = _run_kd_rrt(course, start_pos, goal_pos)
        
        # Run MCTS
        mcts_result = _run_mcts(course, start_pos, goal_pos)
        
        # Store results
        results.append({
            'trial': trial + 1,
            'kd_rrt_converged': kd_result['converged'],
            'kd_rrt_total_action': kd_result['total_action'],
            'kd_rrt_runtime': kd_result['runtime'],
            'kd_rrt_path_length': kd_result['path_length'],
            'kd_rrt_cost': kd_result['cost'],
            'mcts_converged': mcts_result['converged'],
            'mcts_total_action': mcts_result['total_action'],
            'mcts_runtime': mcts_result['runtime'],
            'mcts_path_length': mcts_result['path_length'],
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

