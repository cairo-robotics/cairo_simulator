import numpy as np

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.utils import wrap_to_interval

def parallel_projection_worker(num_samples, sim_context_cls, sim_config, tsr, lazy=False):
    sim_context = sim_context_cls(sim_config, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": True, "use_gui": False})
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    valid_samples = []

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        while len(valid_samples) < num_samples:
            sample = scs.sample()
            if svc.validate(sample):
                q_constrained = project_config(sawyer_robot, tsr, np.array(
                sample), np.array(sample), epsilon=.1, e_step=.25)
                normalized_q_constrained = []
                if q_constrained is not None:
                    for value in q_constrained:
                        normalized_q_constrained.append(
                            wrap_to_interval(value))
                else:
                    continue
                if lazy:
                    print(normalized_q_constrained)
                    valid_samples.append(normalized_q_constrained)
                elif svc.validate(normalized_q_constrained):
                    valid_samples.append(normalized_q_constrained)
    return valid_samples