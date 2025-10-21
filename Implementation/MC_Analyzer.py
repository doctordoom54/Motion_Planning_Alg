import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MonteCarlo_Comparison import compare_algorithms_monte_carlo


def plot_action_vs_trial(df, save_path=None):
    """
    Plot total action vs trial number for both algorithms.
    Shows how action varies across different random courses.
    
    Args:
        df: DataFrame from compare_algorithms_monte_carlo
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    trials = df['trial'].values
    kd_actions = df['kd_rrt_total_action'].values
    mcts_actions = df['mcts_total_action'].values
    
    # Plot individual trials
    ax.plot(trials, kd_actions, 'o-', color='blue', linewidth=2, markersize=8, 
            label='KD-RRT', alpha=0.7)
    ax.plot(trials, mcts_actions, 's-', color='red', linewidth=2, markersize=8,
            label='MCTS', alpha=0.7)
    
    # Mark failed trials with X
    kd_failed = df[df['kd_rrt_converged'] == 0]['trial'].values
    mcts_failed = df[df['mcts_converged'] == 0]['trial'].values
    
    for t in kd_failed:
        ax.plot(t, 0, 'x', color='blue', markersize=15, markeredgewidth=3)
    for t in mcts_failed:
        ax.plot(t, 0, 'x', color='red', markersize=15, markeredgewidth=3)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Total Action', fontsize=12)
    ax.set_title('Total Action vs Trial Number', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_separate_histograms(df, save_path=None):
    """
    Plot separate histograms for KD-RRT and MCTS total actions.
    
    Args:
        df: DataFrame from compare_algorithms_monte_carlo
        save_path: Optional path to save the figure
    """
    # Filter successful trials
    kd_actions = df[df['kd_rrt_converged'] == 1]['kd_rrt_total_action'].dropna()
    mcts_actions = df[df['mcts_converged'] == 1]['mcts_total_action'].dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # KD-RRT Histogram
    ax1.hist(kd_actions, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(kd_actions.mean(), color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {kd_actions.mean():.2f}')
    ax1.set_xlabel('Total Action', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('KD-RRT Total Action Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # MCTS Histogram
    ax2.hist(mcts_actions, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(mcts_actions.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {mcts_actions.mean():.2f}')
    ax2.set_xlabel('Total Action', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('MCTS Total Action Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comprehensive_summary(df, save_path=None):
    """
    Create a comprehensive 2x2 subplot summary of all metrics.
    Best overall visualization combining multiple perspectives.
    
    Args:
        df: DataFrame from compare_algorithms_monte_carlo
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Filter successful trials
    kd_success = df[df['kd_rrt_converged'] == 1]
    mcts_success = df[df['mcts_converged'] == 1]
    
    # 1. Histogram of Total Actions
    ax1 = plt.subplot(2, 2, 1)
    kd_actions = kd_success['kd_rrt_total_action'].dropna()
    mcts_actions = mcts_success['mcts_total_action'].dropna()
    
    if len(kd_actions) > 0 and len(mcts_actions) > 0:
        bins = np.linspace(
            min(kd_actions.min(), mcts_actions.min()),
            max(kd_actions.max(), mcts_actions.max()),
            15
        )
        ax1.hist(kd_actions, bins=bins, alpha=0.6, label='KD-RRT', color='blue', edgecolor='black')
        ax1.hist(mcts_actions, bins=bins, alpha=0.6, label='MCTS', color='red', edgecolor='black')
        ax1.axvline(kd_actions.mean(), color='blue', linestyle='--', linewidth=2)
        ax1.axvline(mcts_actions.mean(), color='red', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Total Action', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Distribution of Total Actions', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Action vs Trial
    ax2 = plt.subplot(2, 2, 2)
    trials = df['trial'].values
    ax2.plot(trials, df['kd_rrt_total_action'].values, 'o-', color='blue', 
             linewidth=2, markersize=6, label='KD-RRT', alpha=0.7)
    ax2.plot(trials, df['mcts_total_action'].values, 's-', color='red',
             linewidth=2, markersize=6, label='MCTS', alpha=0.7)
    ax2.set_xlabel('Trial Number', fontsize=10)
    ax2.set_ylabel('Total Action', fontsize=10)
    ax2.set_title('Total Action vs Trial', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Average
    ax3 = plt.subplot(2, 2, 3)
    kd_cumavg = []
    mcts_cumavg = []
    kd_valid = []
    mcts_valid = []
    
    for i in range(len(trials)):
        if not np.isnan(df['kd_rrt_total_action'].values[i]):
            kd_valid.append(df['kd_rrt_total_action'].values[i])
        if not np.isnan(df['mcts_total_action'].values[i]):
            mcts_valid.append(df['mcts_total_action'].values[i])
        
        kd_cumavg.append(np.mean(kd_valid) if len(kd_valid) > 0 else np.nan)
        mcts_cumavg.append(np.mean(mcts_valid) if len(mcts_valid) > 0 else np.nan)
    
    ax3.plot(trials, kd_cumavg, 'o-', color='blue', linewidth=2.5, markersize=6,
             label='KD-RRT', alpha=0.8)
    ax3.plot(trials, mcts_cumavg, 's-', color='red', linewidth=2.5, markersize=6,
             label='MCTS', alpha=0.8)
    
    if len(kd_cumavg) > 0 and not np.isnan(kd_cumavg[-1]):
        ax3.axhline(kd_cumavg[-1], color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    if len(mcts_cumavg) > 0 and not np.isnan(mcts_cumavg[-1]):
        ax3.axhline(mcts_cumavg[-1], color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax3.set_xlabel('Trial Number', fontsize=10)
    ax3.set_ylabel('Cumulative Average Total Action', fontsize=10)
    ax3.set_title('Cumulative Average Over Trials', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics
    kd_conv_rate = (df['kd_rrt_converged'].sum() / len(df)) * 100
    mcts_conv_rate = (df['mcts_converged'].sum() / len(df)) * 100
    
    stats_text = f"""
    SUMMARY STATISTICS
    {'=' * 50}
    
    Convergence Rates:
      KD-RRT:  {kd_conv_rate:.1f}% ({df['kd_rrt_converged'].sum()}/{len(df)} trials)
      MCTS:    {mcts_conv_rate:.1f}% ({df['mcts_converged'].sum()}/{len(df)} trials)
    
    Total Action (Successful Trials):
      KD-RRT:  Mean = {kd_actions.mean():.2f}, Std = {kd_actions.std():.2f}
      MCTS:    Mean = {mcts_actions.mean():.2f}, Std = {mcts_actions.std():.2f}
    
    Runtime (All Trials):
      KD-RRT:  Mean = {df['kd_rrt_runtime'].mean():.2f}s
      MCTS:    Mean = {df['mcts_runtime'].mean():.2f}s
    
    Path Length (Successful Trials):
      KD-RRT:  Mean = {kd_success['kd_rrt_path_length'].mean():.1f}
      MCTS:    Mean = {mcts_success['mcts_path_length'].mean():.1f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Monte Carlo Analysis Summary ({len(df)} Trials)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_monte_carlo_results(num_trials):
    """
    Main function to run Monte Carlo comparison and generate visualizations.
    
    Args:
        num_trials: Number of trials to run
    
    Returns:
        df: DataFrame with all results
    """
    # Run the comparison
    df = compare_algorithms_monte_carlo(num_trials)
    
    # Generate only the requested plots
    fig1 = plot_comprehensive_summary(df)
    fig2 = plot_action_vs_trial(df)
    fig3 = plot_separate_histograms(df)
    
    plt.show()
    
    return df


if __name__ == '__main__':
    # Run analysis with desired number of trials
    NUM_TRIALS = 35
    
    df = analyze_monte_carlo_results(num_trials=NUM_TRIALS)


