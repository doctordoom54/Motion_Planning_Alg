import numpy as np

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y)
        self.parent = parent
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacle_list, map_bounds, step_size=1.0, goal_sample_rate=0.1, max_iter=500):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list
        self.map_bounds = map_bounds  # (xmin, xmax, ymin, ymax)
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]

    def plan(self):
        """Main planning loop for RRT*"""
        for _ in range(self.max_iter):
            rnd_point = self.sample()
            nearest_node = self.get_nearest_node(rnd_point)
            new_node = self.steer(nearest_node, rnd_point)
            if new_node is None:
                continue
            if not self.collision_path(nearest_node, new_node):
                near_nodes = self.find_near_nodes(new_node)
                self.choose_parent(new_node, near_nodes)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)
        return self.get_final_path()

    def sample(self):
        """Sample a random point in the map (with goal biasing)"""
        x, y = self.start.position
        u, v = self.goal.position
        xmin, xmax, ymin, ymax = self.map_bounds
        while True:
            if np.random.rand() < self.goal_sample_rate:
                center = np.array([u, v])  # Bias toward goal
            else:
                center = np.array([x, y])
            point = np.random.normal(center, 2.0)
            point[0] = np.clip(point[0], xmin, xmax)
            point[1] = np.clip(point[1], ymin, ymax)
            if any(np.allclose(node.position, point) for node in self.node_list):
                continue
            return tuple(point)

    def get_nearest_node(self, point):
        """Find the nearest node in the tree to the sampled point"""
        distances = [np.linalg.norm(np.array(node.position) - np.array(point)) for node in self.node_list]
        nearest_node = self.node_list[np.argmin(distances)]
        return nearest_node

    def steer(self, from_node, to_point):
        """Return a new node in the direction of to_point from from_node"""
        from_pos = np.array(from_node.position)
        to_pos = np.array(to_point)
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None  # Already at the point
        direction = direction / distance  # Normalize direction
    # Move at most step_size toward to_point
        new_pos = from_pos + min(self.step_size, distance) * direction
        new_node = Node(tuple(new_pos), parent = from_node)
        new_node.cost = from_node.cost + np.linalg.norm(new_pos - from_pos)
        return new_node

    def collision(self, node):
        """Check if a node is inside any obstacle."""
        x, y = node.position if isinstance(node, Node) else node
        for ox, oy, c in self.obstacle_list:
            if ox <= x < ox + c and oy <= y < oy + c:
                return True
        return False

    def collision_path(self, from_node, to_node, step_size=0.5):
        """Check if the path from from_node to to_node collides with any obstacle."""
        from_pos = np.array(from_node.position)
        to_pos = np.array(to_node.position)
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return False
        direction = direction / distance
        steps = int(distance / step_size)
        for i in range(steps + 1):
            point = from_pos + i * step_size * direction
            for ox, oy, c in self.obstacle_list:
                if ox <= point[0] < ox + c and oy <= point[1] < oy + c:
                    return True
        return False

    def find_near_nodes(self, new_node):
        """Find nearby nodes for possible rewiring (within a radius)."""
        n = len(self.node_list) + 1
        r = min(50.0 * np.sqrt((np.log(n) / n)), self.step_size * 5)  # radius heuristic
        dists = [np.linalg.norm(np.array(node.position) - np.array(new_node.position)) for node in self.node_list]
        near_nodes = [self.node_list[i] for i, d in enumerate(dists) if d <= r]
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        """Choose the best parent for new_node from near_nodes (lowest cost, collision-free)."""
        min_cost = new_node.cost
        best_parent = new_node.parent
        for node in near_nodes:
            if not self.collision_path(node, new_node):
                cost = node.cost + np.linalg.norm(np.array(node.position) - np.array(new_node.position))
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, near_nodes):
        """Rewire the tree if a shorter path is found through new_node."""
        for node in near_nodes:
            if node == new_node.parent:
                continue
            if not self.collision_path(new_node, node):
                new_cost = new_node.cost + np.linalg.norm(np.array(node.position) - np.array(new_node.position))
                if new_cost < node.cost:
                    node.parent = new_node
                    node.cost = new_cost

    def get_final_path(self):
        """Trace back from the node closest to the goal to the start to get the final path."""
        # Find node closest to goal
        dists = [np.linalg.norm(np.array(node.position) - np.array(self.goal.position)) for node in self.node_list]
        min_idx = np.argmin(dists)
        node = self.node_list[min_idx]
        path = [node.position]
        while node.parent is not None:
            node = node.parent
            path.append(node.position)
        path.reverse()
        return path
