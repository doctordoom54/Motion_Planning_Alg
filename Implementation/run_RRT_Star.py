import sys
import os

# Add the parent directory to Python path so we can import from Obstacle Course
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obstacle_course import ObstacleCourse
import numpy as np
import scipy
from visualize_course import viz


viz(250, 250, 20, 15)


#Course =course = ObstacleCourse(
###        n=width, 
   #     m=height, 
   #     num_obstacles=num_obstacles,
    #    obstacle_size=obstacle_size
    #)
#start, goal = course.get_start_goal()
