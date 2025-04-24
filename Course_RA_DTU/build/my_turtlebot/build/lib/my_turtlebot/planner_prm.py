"""
Implement a probailistic roadmap method.

- Subscribe to the map topic and use that map for collision checks.
- Assume constant cost value for all nodes.
- Use any nearest neighbor implementation.
- Create a function that takes as input a number of randomly samped node positions and returns the graph G.
"""
import rclpy
import numpy as np
import math
from collections import defaultdict
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from skimage.morphology import dilation, disk
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(8)

SUB_NAME = "prm"
SUB_MSG = OccupancyGrid
SUB_TOPIC = "/map"


class PRM(Node):
    def __init__(self):
        super().__init__(SUB_NAME)
        map_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.map_sub = self.create_subscription(
            SUB_MSG, SUB_TOPIC, self.map_callback, map_profile
        )
        
        # PRM parameters
        self.num_nodes = 300
        self.num_neighbors = 4  # Number of nearest neighbors
        self.weight = 1         # Constant cost value

        # Variables to create map
        self.nodes = []
        self.edges = defaultdict(list)

        self.start = [50, 30]
        self.start_idx = None

        self.goal = [270, 40]
        self.goal_idx = None

    def map_callback(self, msg: SUB_MSG):
        print("[PRM] Map received.")
        # Process the received message
        self.map = msg
        self.map_data = msg.data        # contains the original map_data
        self.map_info = msg.info
        self.map_metadata = msg.info
        self.map_resolution = self.map_metadata.resolution
        self.map_width = self.map_metadata.width
        self.map_height = self.map_metadata.height
        self.map_origin = self.map_metadata.origin

        self.dilate_map(self.map_data, clearance=0.2)  # for path clearance

        # Create the roadmap the first time the map is received
        while not self.nodes:
            print("[PRM] Creating roadmap...")
            self.generate_random_nodes()
            self.connect_k_nearest_neighbors()  # choose either this
            # self.create_k_nearest_edges()     # or this
            self.add_start_and_goal(self.start, self.goal)
            distances, path = dijkstra(self.edges, self.start_idx, self.goal_idx)
            self.plot(path)

    def generate_random_nodes(self):
        """
        Uniformly generate random nodes in the map space.
        """
        print("[PRM] Generating random nodes...")
        for _ in range(self.num_nodes):
            collides = True
            while collides:
                x = np.random.randint(0, self.map_width)
                y = np.random.randint(0, self.map_height)
                # Check if node collides with obstacles
                node = (x, y)
                if not self.node_collides_obstacle(node):
                    collides = False
            self.nodes.append(node)
        
    def node_collides_obstacle(self, node):
        """
        Check if the node collides with an obstacle in the map.
        """
        x, y = node
        if self.dilated_map[y * self.map_width + x] > 0:
            return True
        return False
    
    def path_collides_obstacle(self, node1:tuple, node2:tuple):
        """
        Check if the path between two nodes collides with an obstacle 
        in the map using Bresenham's line algorithm.
        """
        path = bresenham(node1, node2)
        for point in path:
            if self.dilated_map[point[1] * self.map_width + point[0]] > 0:
                return True
        return False
    
    def sorted_neighbors(self, node):
        """
        Return a list of sorted neighbors for a given node.

        Returns:
            List[Tuple[float, int]] : List of distances and indices of the neighbors 
                                      of length n
        """
        distances = []
        for i, other_node in enumerate(self.nodes):
            if node != other_node:
                dist = math.sqrt((node[0] - other_node[0]) ** 2 + (node[1] - other_node[1]) ** 2)
                distances.append((dist, i))
        return sorted(distances)
    
    def find_k_nearest_neighbors(self, nodes, k):
        """
        Finds indices of the K nearest neighbors for each node.

        Args:
            nodes (List[Tuple[float, float]]): List of the nodes positions
            k (int): Number of nearest nodes to connect to

        Returns:
            dict: dictionary where the key-value pairs are:  node_idx: [neighbor_1_idx, neighbor_2_idx, ..., neighbor_k_idx]
        """
        node_neighbors = defaultdict(list)

        for i, node1 in enumerate(nodes):
            k_nearest = self.sorted_neighbors(node1)[:k]
            for _, neighbor_idx in k_nearest:
                node_neighbors[i].append(neighbor_idx)
        return node_neighbors

    def connect_k_nearest_neighbors(self):
        """
        Finds indices of the K nearest neighbors for each node.

        Args:
            nodes (List[Tuple[float, float]]): List of the nodes positions
            k (int): Number of nearest nodes to connect to
        """
        print("[PRM] Creating edges...")
        for i, node1 in enumerate(self.nodes):
            k_nearest = self.sorted_neighbors(node1)[:self.num_neighbors]
            for _, neighbor_idx in k_nearest:
                # If path is free, calculate distance and create the edge
                if not self.path_collides_obstacle(node1, self.nodes[neighbor_idx]):
                    dist = self.weight
                    if (neighbor_idx, dist) not in self.edges[i]:
                        self.edges[i].append((neighbor_idx, dist))
                    if (i, dist) not in self.edges[neighbor_idx]:
                        self.edges[neighbor_idx].append((i, dist))
    
    def create_k_nearest_edges(self):
        """
        Create k edges between the nearest neighbors with obstacle free paths.
        """
        print("[PRM] Creating edges...")
        node_neighbors = defaultdict(list)
        for i, node1 in enumerate(self.nodes):
            sorted_neighbors = self.sorted_neighbors(node1)

            # while there are less than k neighbors & unvisited neighbors remaining
            while len(node_neighbors[i]) < self.num_neighbors and sorted_neighbors:
                neighbor_idx = sorted_neighbors.pop(0)[1]
                # If path is free, calculate distance and create the edge
                if not self.path_collides_obstacle(node1, self.nodes[neighbor_idx]):
                    dist = self.weight
                    if (neighbor_idx, dist) not in self.edges[i]:
                        self.edges[i].append((neighbor_idx, dist))
                    if (i, dist) not in self.edges[neighbor_idx]:
                        self.edges[neighbor_idx].append((i, dist))
                    node_neighbors[i].append(neighbor_idx)


    def add_start_and_goal(self, start, goal):
        """
        Add start and goal points to the roadmap.

        Args:
            start (Tuple[float, float]): Start position
            goal (Tuple[float, float]): Goal position
        """
        print("[PRM] Adding start and goal nodes...")
        if self.node_collides_obstacle(start):
            raise ValueError("Start node collides with an obstacle.")
        if self.node_collides_obstacle(goal):
            raise ValueError("Goal node collides with an obstacle.")
        
        # Set start and goal positions, and add them to the nodes
        # self.start, self.goal = start, goal

        self.start_idx = len(self.nodes)
        self.nodes.append(start)

        self.goal_idx = len(self.nodes)
        self.nodes.append(goal)

        # Find nearest node
        neighbours = self.find_k_nearest_neighbors(self.nodes, k=1)

        # Get the closest (first) neighbour to the star and goal positions
        goal_neighbour_idx = neighbours[self.goal_idx][0]
        start_neighbour_idx = neighbours[self.start_idx][0]

        # Compute distances
        d_goal = self.weight
        d_start = self.weight

        self.edges[goal_neighbour_idx].append((self.goal_idx, d_goal))
        self.edges[start_neighbour_idx].append((self.start_idx, d_start))
        self.edges[self.goal_idx].append((goal_neighbour_idx, d_goal))
        self.edges[self.start_idx].append((start_neighbour_idx, d_start))
    
    def dilate_map(self, map_data, clearance):
        """
        Dilate the obstacles on the map to replicate path clearance.

        Args:
            map_data (List[int]): The map data representing obstacles.
            clearance (float): The clearance value to add around obstacles in metres.

        Returns:
            List[int]: The dilated map data.
        """
        disk_radius = int(clearance / self.map_resolution)

        # Convert map_data to a 2D array
        map_array = np.array(map_data).reshape((self.map_height, self.map_width))
        dilated_map = dilation(map_array, disk(disk_radius))

        # Convert the dilated map back to a 1D array 
        self.dilated_map = dilated_map.ravel().tolist()

    def plot(self, path=None):
        print("[PRM] Plotting...")
        plt.figure(dpi=120)
        
        # Plot edges
        for node_idx, edges in self.edges.items():
            for edge in edges:
                node1 = self.nodes[node_idx]
                node2 = self.nodes[edge[0]]
                plt.plot([node1[0], node2[0]], [node1[1], node2[1]], "b-", label="edges")

        # Plot path
        if path:
            for i in range(len(path)-1):
                node1 = self.nodes[path[i]]
                node2 = self.nodes[path[i+1]]
                plt.plot([node1[0], node2[0]], [node1[1], node2[1]], "r-", linewidth=2)

        # Plot nodes
        for i, node in enumerate(self.nodes):
            plt.plot(node[0], node[1], "ro", markersize=3)
            # plt.text(node[0], node[1], str(i), ha='center', va='center')  # index
        
        # Plot clearance (takes a while)
        x_list, y_list = [], []
        for i,value in enumerate(self.dilated_map):
            if value > 0:
                x_list.append(i % self.map_width)
                y_list.append(i // self.map_width)
        plt.scatter(x_list, y_list, color='yellow', s=1)

        # Plot obstacles
        x_list, y_list = [], []
        for i,value in enumerate(self.map_data):
            if value > 0:
                x_list.append(i % self.map_width)
                y_list.append(i // self.map_width)
        plt.scatter(x_list, y_list, color='black', s=1)
        
        # Plot start and goal
        plt.plot(self.start[0], self.start[1], color='green', marker='o', markersize=2)
        plt.plot(self.goal[0], self.goal[1], color='#fca103', marker='o', markersize=2)
        
        # Prepare custom legend
        lg_handles = [
                    Line2D([0], [0], color='black', marker='o', markersize=10, label='Obstacles'),
                    Line2D([0], [0], color='yellow', marker='o', markersize=10, label='Clearance'),
                    Line2D([0], [0], color='red', marker='o', markersize=10, label='Nodes'),
                    Line2D([0], [0], color='green', marker='o', markersize=10, label='Start'),
                    Line2D([0], [0], color='#fca103', marker='o', markersize=10, label='Goal'),
                    Line2D([0], [0], color='blue', label='Edges'),
                    Line2D([0], [0], color='red', linewidth=2, label='Path'),
                    ]
        # plt.legend(loc="upper right", handles=lg_handles, bbox_to_anchor=(1.4, 1.))
        # TODO: i can't get the legend to be displayed nicely on the side without affecting the grid

        plt.title(f"Probabilistic Roadmap, n={len(self.nodes)} nodes, k={self.num_neighbors} neighbors")
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.tight_layout()
        plt.grid(True)
        plt.axis('scaled')
        plt.show()
        print("Plotting done.")


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end including.

    (original from roguebasin.com)
    >> points1 = bresenham((4, 4), (6, 10))
    >> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def dijkstra(edges, start_idx, goal_idx):
    '''
    Use dijkstra algorithm to find the shortest path from start to goal.

    Args:
        edges (dict): dictionary where the key-value pairs are:  node_idx: [(neighbor_idx, distance), ...]
        start_idx (int): index of the start node
        goal_idx (int): index of the goal node

    Returns:
        dict: dictionary where the key-value pairs are:  node_idx: distance
        path: list of the node indexes in the shortest path
    '''
    print(f"[Dijkstra] Finding path from {start_idx} to {goal_idx} using Dijkstra algorithm.")

    class Node:
        def __init__(self, idx):
            self.idx = idx
            self.parent = None

    # Initialize the distance to all nodes to be infinity
    distances = {}
    nodes = {}
    for node in edges:
        distances[node] = float("inf")
        nodes[node] = Node(node)

    # Initialize start node
    distances[start_idx] = 0
    nodes[start_idx].parent = start_idx

    queue = [(start_idx, 0)]  # priority queue: store nodes and their distances
    visited = set()           # store visited nodes

    # Iterate through the queue
    while queue:
        current_node, current_distance = queue.pop(0)

        if current_node in visited:
            continue

        # Check if the current node is the goal node
        if current_node == goal_idx:
            print(f"[Dijkstra] Visited {len(visited)} nodes")
            print(f"[Dijkstra] Goal node {goal_idx} found with distance {current_distance}")

            # Reconstruct the path
            path = []
            while current_node != start_idx:
                path.append(current_node)
                current_node = nodes[current_node].parent
            path.append(start_idx)
            path.reverse()
            print(f"[Dijkstra] Path: {path}")
            break

        # Explore the neighbors of the current node
        for neighbor, distance in edges[current_node]:
            # Relax the current node if a lower distance estimate is found
            new_estimate = current_distance + distance
            if new_estimate < distances[neighbor]:
                distances[neighbor] = new_estimate
                nodes[neighbor].parent = current_node
            queue.append((neighbor, distances[neighbor]))

        visited.add(current_node)

        # Sort the queue based on the distances
        queue.sort(key=lambda x: x[1])
    
    return distances, path


def main():   
    rclpy.init()
    prm = PRM()
    rclpy.spin(prm)
    rclpy.shutdown()    

if __name__ == "__main__":
    main()
