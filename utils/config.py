
from dataclasses import dataclass, field

from typing import List, Dict, Optional, Literal



@dataclass
class Configuration:
    """Main configuration container for VRPSDSD solver."""
    
    # Problem parameters
    vehicle_capacity: float
    distance_metric: str = "euclidean"
    
    # Cost computation
    cost_method: Literal["exact", "sampling"] = "exact"
    recourse_policy: str = "paired_vehicle"
    
    # Sampling configuration
    sampling_num_samples: int = 1000
    sampling_random_seed: Optional[int] = None
    sampling_parallel: bool = False
    sampling_num_threads: Optional[int] = None
    sampling_variance_reduction: List[str] = field(default_factory=list)
    
    # ALNS parameters
    alns_iterations: int = 1000
    alns_segment_length: int = 100
    alns_start_temperature: float = 100.0
    alns_cooling_rate: float = 0.9995
    
    # Operator parameters
    removal_min: int = 1
    removal_max: int = 10
    insertion_min: int = 1
    insertion_max: int = 10
    
    # Weight adaptation
    weight_update_decay: float = 0.8
    reaction_factor: float = 0.3
    score_increment: Dict[str, float] = field(default_factory=lambda: {
        "new_best": 10.0,
        "improving": 5.0,
        "accepted": 2.0,
        "rejected": 1.0
    })
    
    # Output and logging
    verbose: bool = True
    log_frequency: int = 100
    save_solutions: bool = False
    output_dir: str = "results"