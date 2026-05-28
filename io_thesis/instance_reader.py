from core.instance import Node, ProblemInstance


def scale_demands(nodes, Q, l, u):
    customers = [n for n in nodes if not n.is_depot]
    if not customers:
        return
    min_d = min(n.mean_demand for n in customers)
    max_d = max(n.mean_demand for n in customers)
    for node in customers:
        if max_d > min_d:
            node.mean_demand = (
                l * Q
                + (u - l) * (node.mean_demand - min_d) / (max_d - min_d) * Q
            )
        else:
            node.mean_demand = (l + u) / 2 * Q


def read_solomon_instance(
    filepath: str,
    vehicle_capacity: float = 70.0,
    demand_scale: tuple = None,
    num_customers: int = None,
) -> ProblemInstance:
    """Read Solomon instance file and return ProblemInstance with given vehicle capacity.
    Ignores time windows and service times; uses only coordinates and demands.
    If num_customers is given, only the first num_customers customer rows are used (depot always included).
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

    customers_read = 0
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
        if not is_depot:
            customers_read += 1
            if num_customers is not None and customers_read >= num_customers:
                break
    
    if demand_scale is not None:
        l, u = demand_scale
        scale_demands(nodes, vehicle_capacity, l, u)

    return ProblemInstance(nodes, vehicle_capacity=vehicle_capacity)


def read_cvrplib_instance(
    filepath: str,
    vehicle_capacity: float = 70.0,
    demand_scale: tuple = None,
) -> ProblemInstance:
    """Read CVRPLIB/Augerat format (.vrp) instance. Node 1 is treated as depot."""
    nodes = []
    coords = {}
    demands = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line == 'EOF':
            continue
        if line.startswith('NODE_COORD_SECTION'):
            section = 'coords'
            continue
        elif line.startswith('DEMAND_SECTION'):
            section = 'demands'
            continue
        elif line.startswith('DEPOT_SECTION'):
            section = 'depot'
            continue
        elif ':' in line:
            section = None
            continue

        if section == 'coords':
            parts = line.split()
            node_id = int(parts[0])
            coords[node_id] = (float(parts[1]), float(parts[2]))
        elif section == 'demands':
            parts = line.split()
            node_id = int(parts[0])
            demands[node_id] = float(parts[1])

    for node_id in sorted(coords.keys()):
        x, y = coords[node_id]
        demand = demands.get(node_id, 0.0)
        is_depot = (node_id == 1)
        nodes.append(Node(id=node_id, x=x, y=y, mean_demand=demand,
                          is_depot=is_depot, is_split=False, alpha=1.0))

    if demand_scale is not None:
        l, u = demand_scale
        scale_demands(nodes, vehicle_capacity, l, u)

    return ProblemInstance(nodes, vehicle_capacity=vehicle_capacity)