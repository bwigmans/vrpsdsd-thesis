from typing import List, Optional
import numpy as np
from scipy.stats import poisson


class Node:
    def __init__(self, id: int, x: float, y: float, mean_demand: float, is_depot: bool = False, is_split: bool = False, alpha =1.0):
        """Node with coordinates and Poisson demand parameter."""
        self.id = id
        self.x = x
        self.y = y
        self.demand_lambda = mean_demand
        self.is_split = is_split
        self.is_depot = is_depot
        self.alpha = alpha  # fraction for split deliveries (if is_split=True)
        
        
    
    def distance_to(self, other: 'Node') -> float:
        """Euclidean distance to another node."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class ProblemInstance:
    def __init__(self, nodes: List[Node], vehicle_capacity: float, 
                 distance_matrix: Optional[np.ndarray] = None, 
                 distance_metric: str = 'euclidean'):
        """Initialize problem instance."""
        self.nodes = nodes
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.distance_metric = distance_metric
        
    
    def _compute_distance_matrix(self, metric: str = 'euclidean') -> np.ndarray:
        """Compute distance matrix from node coordinates."""
        n = len(self.nodes)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'euclidean':
                    dist = self.nodes[i].distance_to(self.nodes[j])
                else:
                    raise ValueError(f"Unsupported distance metric: {metric}")
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def get_distance(self, node_i: Node, node_j: Node) -> float:
        """Get distance between two nodes."""
        if self.distance_matrix is not None:
            return self.distance_matrix[node_i.id, node_j.id]
        else:
            return node_i.distance_to(node_j)

    def get_demand_distribution(self, node: Node):
        """Get Poisson distribution for node's demand."""
        return poisson(mu=node.demand_lambda)
       
    
    def get_expected_demand(self, node: Node) -> float:
        """Get expected demand for node (lambda parameter)."""
        if node.is_split :
            return node.demand_lambda * node.alpha
        return node.demand_lambda
    
    def validate(self) -> bool:
        """Validate instance consistency."""
        if not self.nodes:
            raise ValueError("Instance must have at least one node.")
        if self.vehicle_capacity <= 0:
            raise ValueError("Vehicle capacity must be positive.")
        if self.distance_matrix is not None:
            n = len(self.nodes)
            if self.distance_matrix.shape != (n, n):
                raise ValueError("Distance matrix shape must match number of nodes.")
        return True
    


if __name__ == "__main__":
    # Create nodes: depot at (0,0) with zero demand, and two customer nodes
    depot = Node(id=0, x=0.0, y=0.0, mean_demand=0.0, is_depot=True)
    customer1 = Node(id=1, x=3.0, y=4.0, mean_demand=2.5)
    customer2 = Node(id=2, x=6.0, y=8.0, mean_demand=1.2)
    
    nodes = [depot, customer1, customer2]
    vehicle_capacity = 10.0
    
    # Create problem instance
    instance = ProblemInstance(nodes, vehicle_capacity)
    
    # Compute and print distance matrix
    dist_matrix = instance._compute_distance_matrix()
    print("Distance matrix:")
    print(dist_matrix)
    print()
    
    # Test distance retrieval
    print(f"Distance from depot to customer1: {instance.get_distance(depot, customer1):.2f}")
    print(f"Distance from customer1 to customer2: {instance.get_distance(customer1, customer2):.2f}")
    print()
    
    # Validate instance
    try:
        instance.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed: {e}")
    print()
    
    # Expected demands
    print("Expected demands:")
    for node in nodes:
        print(f"  Node {node.id} (depot={node.is_depot}): lambda = {instance.get_expected_demand(node)}")
    print()
    
    # Demand distribution example
    print("Poisson demand samples for customer1 (lambda=2.5):")
    dist = instance.get_demand_distribution(customer1)
    samples = dist.rvs(size=10)
    print(f"  Samples: {samples}")


