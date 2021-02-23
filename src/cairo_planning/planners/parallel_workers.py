import numpy as np

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval


def parallel_sample_worker(num_samples, sim_context_cls, sim_config):
    sim_context = sim_context_cls(sim_config, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": True, "use_gui": False})
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()

    # Disabled collisions during planning with certain exclusions in place.
    # collision_exclusions = sim_context.get_collision_exclusions()
    collision_exclusions = {}

    with DisabledCollisionsContext(sim, **collision_exclusions):
        valid_samples = []
        count = 0
        while count < num_samples:
            q_rand = np.array(state_space.sample())
            if svc.validate(q_rand):
                valid_samples.append(q_rand)
                count += 1
    return valid_samples


def parallel_connect_worker(batches, interp_fn, distance_fn, sim_context_cls, sim_config):
    sim_context = sim_context_cls(sim_config, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": True, "use_gui": False})
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    _ = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()

    # Disabled collisions during planning with certain exclusions in place.
    # collision_exclusions = sim_context.get_collision_exclusions()
    collision_exclusions = {}

    with DisabledCollisionsContext(sim, **collision_exclusions):
        connections = []
        for batch in batches:
            q_sample = batch[0]
            neighbors = batch[1]
            for q_near in neighbors:
                local_path = interp_fn(
                    np.array(q_near), np.array(q_sample), steps=10)
                valid = subdivision_evaluate(svc.validate, local_path)
                if valid:
                    connections.append(
                        [q_near, q_sample, distance_fn(local_path)])
        return connections

def parallel_projection_worker(num_samples, sim_context_cls, sim_config, tsr):
    sim_context = sim_context_cls(sim_config, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": True, "use_gui": False})
    sim = sim_context.get_sim_instance()
    constrained_state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling {} samples".format(num_samples))
        count = 0
        valid_samples = []
        while count <= num_samples:
            # start_time = timer()
            q_rand = np.array(constrained_state_space.sample())
            if np.any(q_rand):
                if svc.validate(q_rand):
                    if count % int(num_samples / 4) == 0:
                        print("{} valid samples...".format(count))
                    valid_samples.append(q_rand)
                    count += 1
                    # sampling_times.append(timer() - start_time)
            # print(sum(sampling_times) / len(sampling_times))
        return valid_samples