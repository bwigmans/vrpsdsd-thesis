from core.instance import Node, ProblemInstance


def read_solomon_instance(filepath: str, vehicle_capacity: float = 70.0) -> ProblemInstance:
    """Read Solomon instance file and return ProblemInstance with given vehicle capacity.
    Ignores time windows and service times; uses only coordinates and demands.
    """
    nodes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the CUSTOMER section
    customer_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('CUSTOMER'):
            customer_start = i + 2  # skip the header line
            break
    
    if customer_start is None:
        raise ValueError("No CUSTOMER section found in file")
    
    for line in lines[customer_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        # Columns: CustNo, X, Y, Demand, ReadyTime, DueDate, ServiceTime
        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = float(parts[3])
        is_depot = (cust_id == 0)
        # For VRPSDSD, we ignore time windows and service time
        node = Node(id=cust_id, x=x, y=y, mean_demand=demand,
                    is_depot=is_depot, is_split=False, alpha=1.0)
        nodes.append(node)
    
    return ProblemInstance(nodes, vehicle_capacity=vehicle_capacity)