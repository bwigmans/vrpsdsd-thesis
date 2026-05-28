import matplotlib.pyplot as plt
import numpy as np
from typing import List
from core.solution import Solution
from core.instance import Node


class SolutionVisualizer:
    def __init__(self, solution: Solution, instance=None):
        self.solution = solution
        self.instance = instance or solution.routes[0].instance

    def plot(
        self,
        show_labels: bool = True,
        show_route_numbers: bool = True,
        show_capacity: bool = False,
        capacity_mode: str = "planned",
        capacity_seed: int = 42,
        show_only_split_routes: bool = False,
        figsize=(10, 8),
        save_path: str = None,
    ):
        """Plot all routes with different colors."""
        plt.figure(figsize=figsize)

        # Plot depot
        depot = self.instance.nodes[0]
        plt.scatter(depot.x, depot.y, c='red', s=150, marker='s', label='Depot', zorder=5)

        # Plot customers (base instance)
        customers = [n for n in self.instance.nodes if not n.is_depot]
        for cust in customers:
            color = 'blue' if not cust.is_split else 'orange'
            plt.scatter(cust.x, cust.y, c=color, s=50, alpha=0.8)
            if show_labels:
                plt.annotate(
                    str(cust.id),
                    (cust.x, cust.y),
                    fontsize=8,
                    ha='center',
                    va='bottom',
                )

        # Overlay split nodes from the solution (may not exist in instance list)
        split_nodes = [n for r in self.solution.routes for n in r.nodes if n.is_split]
        if split_nodes:
            plt.scatter(
                [n.x for n in split_nodes],
                [n.y for n in split_nodes],
                c='orange',
                s=90,
                marker='^',
                edgecolors='black',
                linewidths=0.6,
                label='Split nodes',
                zorder=6,
            )
            if show_labels:
                for node in split_nodes:
                    label = f"{node.id} (a={node.alpha:.2f})"
                    plt.annotate(label, (node.x, node.y), fontsize=8, ha='center', va='bottom')

        # Plot routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.solution.routes)))
        split_routes = {
            idx for idx, route in enumerate(self.solution.routes)
            if any(node.is_split for node in route.nodes)
        }
        for idx, route in enumerate(self.solution.routes):
            is_split_route = idx in split_routes
            if show_only_split_routes and not is_split_route:
                continue
            xs = [n.x for n in route.nodes]
            ys = [n.y for n in route.nodes]
            plt.plot(
                xs,
                ys,
                color=colors[idx],
                linewidth=3 if is_split_route else 2,
                marker='o',
                markersize=7 if is_split_route else 6,
                label=(
                    f'Route {idx+1} (split)'
                    if show_route_numbers and is_split_route
                    else (f'Route {idx+1}' if show_route_numbers else '_nolegend_')
                ),
            )
            # Mark start of route
            if route.nodes[0].is_depot and len(route.nodes) > 1:
                first_cust = route.nodes[1]
                plt.annotate(f'R{idx+1}', (first_cust.x, first_cust.y),
                             fontsize=9, fontweight='bold', xytext=(5, 5),
                             textcoords='offset points', color=colors[idx])

            if show_capacity:
                remaining = self.instance.vehicle_capacity
                rng = np.random.default_rng(capacity_seed + idx)
                for node in route.nodes[1:]:
                    if node.is_depot:
                        continue
                    label_parts = [f"{remaining:.1f}"]
                    if node.is_split:
                        label_parts.append(f"a={node.alpha:.2f}")
                    label = " ".join(label_parts)
                    plt.annotate(
                        label,
                        (node.x, node.y),
                        fontsize=7,
                        ha='center',
                        va='top',
                        xytext=(0, -6),
                        textcoords='offset points',
                        color=colors[idx],
                    )

                    if capacity_mode == "simulated":
                        dist = self.instance.get_demand_distribution(node)
                        demand_total = float(dist.rvs(random_state=rng))
                        planned = demand_total * node.alpha if node.is_split else demand_total
                        if planned > remaining + 1e-9:
                            leftover = planned - remaining
                            remaining = max(0.0, self.instance.vehicle_capacity - leftover)
                        elif abs(planned - remaining) < 1e-9:
                            remaining = self.instance.vehicle_capacity
                        else:
                            remaining -= planned
                    else:
                        planned = route._planned_demand(node)
                        remaining = max(0.0, remaining - planned)

        plt.legend(loc='best', fontsize=8)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_route(self, route_index: int, show_labels=True, figsize=(8, 6)):
        """Plot a single route."""
        route = self.solution.routes[route_index]
        plt.figure(figsize=figsize)

        xs = [n.x for n in route.nodes]
        ys = [n.y for n in route.nodes]
        plt.plot(xs, ys, 'b-o', linewidth=2, markersize=8)

        # Mark depot and customers
        for i, node in enumerate(route.nodes):
            if node.is_depot:
                plt.scatter(node.x, node.y, c='red', s=120, marker='s', label='Depot' if i == 0 else '')
            else:
                color = 'orange' if node.is_split else 'blue'
                plt.scatter(node.x, node.y, c=color, s=60)
            if show_labels:
                label = f"{node.id}" + (f" (α={node.alpha:.2f})" if node.is_split else "")
                plt.annotate(label, (node.x, node.y), fontsize=8, ha='center', va='bottom')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Route {route_index+1} (Cost: {route.travel_cost():.2f}, Load: {route.expected_load():.2f})')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    def plot_route_failure_risk(self, route_index: int, show_labels=True, figsize=(10, 8), save_path=None):
        """Plot a single route with nodes colored by failure probability (red = high risk)."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        route = self.solution.routes[route_index]
        probs = route.failure_probabilities()  # per customer in order
        customers = [n for n in route.nodes if not n.is_depot]
        # Map node id -> failure probability in this route
        node_risk = {node.id: prob for node, prob in zip(customers, probs)}

        plt.figure(figsize=figsize)

        # Depot
        depot = self.instance.nodes[0]
        plt.scatter(depot.x, depot.y, c='red', s=150, marker='s', label='Depot', zorder=5)

        # Plot customers with color based on risk
        if node_risk:
            norm = Normalize(vmin=0, vmax=max(node_risk.values()))
        else:
            norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.RdYlGn_r

        for node in self.instance.nodes:
            if node.is_depot:
                continue
            risk = node_risk.get(node.id, 0)
            color = cmap(norm(risk))
            plt.scatter(node.x, node.y, c=[color], s=80, edgecolor='black', linewidth=0.5, zorder=3)
            if show_labels:
                label = f"{node.id}\n({risk:.2%})" if risk > 0 else str(node.id)
                plt.annotate(label, (node.x, node.y), fontsize=7, ha='center', va='bottom')

        # Draw the route line
        xs = [n.x for n in route.nodes]
        ys = [n.y for n in route.nodes]
        plt.plot(xs, ys, color='blue', linewidth=2, alpha=0.8, zorder=2, label=f'Route {route_index+1}')

        # Colorbar
        if node_risk:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
            cbar.set_label('Failure probability')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Failure Risk Map – Route {route_index+1}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_split_node_routes(self, split_node_id: int, figsize=(12, 6)):
        """
        Plot the two routes that share a split node.
        Left: route maps with failure probability colors.
        Right: bar charts of failure probabilities along each route.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Find the two routes containing this node
        routes = []
        for route in self.solution.routes:
            if any(n.id == split_node_id for n in route.nodes):
                routes.append(route)
        if len(routes) != 2:
            raise ValueError(f"Node {split_node_id} appears in {len(routes)} routes, expected 2.")

        # Compute failure probabilities for each route
        probs1 = routes[0].failure_probabilities()
        probs2 = routes[1].failure_probabilities()
        # Get the customer nodes in order (excluding depot)
        cust1 = [n for n in routes[0].nodes if not n.is_depot]
        cust2 = [n for n in routes[1].nodes if not n.is_depot]

        # Determine global probability range for consistent coloring
        all_probs = probs1 + probs2
        vmin, vmax = 0, max(all_probs) if all_probs else 1
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlGn_r

        # Create figure with 2 rows, 2 columns
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_map1, ax_bar1 = axes[0, 0], axes[0, 1]
        ax_map2, ax_bar2 = axes[1, 0], axes[1, 1]

        # Helper to draw a route map with colored nodes
        def draw_map(ax, route, probs, cust_nodes):
            depot = self.instance.nodes[0]
            ax.scatter(depot.x, depot.y, c='red', s=100, marker='s', label='Depot', zorder=5)
            xs = [n.x for n in route.nodes]
            ys = [n.y for n in route.nodes]
            ax.plot(xs, ys, 'gray', linewidth=1.5, alpha=0.6, zorder=1)
            for node, prob in zip(cust_nodes, probs):
                color = cmap(norm(prob))
                ax.scatter(node.x, node.y, c=[color], s=70, edgecolor='black', linewidth=0.5, zorder=3)
                label = f"{node.id}\n({prob:.2%})" if prob > 0 else str(node.id)
                ax.annotate(label, (node.x, node.y), fontsize=8, ha='center', va='bottom')
            ax.set_title(f"Route {route.nodes[0].id} → ... → {route.nodes[-1].id}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        # Draw maps
        draw_map(ax_map1, routes[0], probs1, cust1)
        draw_map(ax_map2, routes[1], probs2, cust2)

        # Draw bar charts
        ax_bar1.bar(range(len(probs1)), probs1, color='steelblue', alpha=0.7)
        ax_bar1.set_xticks(range(len(cust1)))
        ax_bar1.set_xticklabels([f"{n.id}" for n in cust1], rotation=45)
        ax_bar1.set_ylabel('Failure probability')
        ax_bar1.set_title('Failure probability per customer')
        ax_bar1.grid(True, alpha=0.3)

        ax_bar2.bar(range(len(probs2)), probs2, color='steelblue', alpha=0.7)
        ax_bar2.set_xticks(range(len(cust2)))
        ax_bar2.set_xticklabels([f"{n.id}" for n in cust2], rotation=45)
        ax_bar2.set_ylabel('Failure probability')
        ax_bar2.set_title('Failure probability per customer')
        ax_bar2.grid(True, alpha=0.3)

        # Add a colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label('Failure probability')

        plt.tight_layout()
        plt.show()