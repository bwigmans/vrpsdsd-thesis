
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
@dataclass
class Configuration:
    """Main configuration container for VRPSDSD solver."""
    
    # Problem parameters
    vehicle_capacity: float
    distance_metric: str = "euclidean"
    
    # Cost computation
    cost_method: Literal["exact", "sampling"] = "sampling"
    recourse_policy: str = "paired_vehicle"

    # Sampling configuration — only used when cost_method == "sampling"
    operator_num_samples: int = 50
    evaluation_num_samples: int = 500

    # Random seed
    seed: Optional[int] = None
    
    # ALNS parameters
    alns_iterations: int = 1000
    alns_segment_length: int = 50

    # RRT acceptance criterion
    rrt_deviation_factor: float = 0.01
    
    # Operator parameters
    removal_min: int = 1
    removal_max: int = 10
    # Weight adaptation
    weight_update_decay: float = 0.1
    score_increment: Dict[str, float] = field(default_factory=lambda: {
        "new_best": 30.0,
        "improving": 10.0,
        "accepted": 6.0,
    })
    
    # Path to precomputed demand sample bank (.npz)
    sample_bank_path: Optional[str] = None

    # EC insertion operators (GreedyInsertionEC + RegretInsertionEC)
    use_ec_operators: bool = False

    # Post-processing: find best split after ALNS loop completes
    find_split_post: bool = False

    # Alpha policy for SplitInsertion
    alpha_policy: str = "lei"
    alpha_grid: Optional[List[float]] = None
    alpha_reoptimize: bool = False

    # Post-processing restart: lock split nodes from removal
    lock_splits: bool = False

    # Output and logging
    verbose: bool = True
    log_frequency: int = 100
    save_solutions: bool = False
    output_dir: str = "results"