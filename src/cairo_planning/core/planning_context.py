from abc import ABC, abstractmethod

from cairo_planning.geometric.state_space import SawyerConfigurationSpace

class AbstractPlanningContext(ABC):

    @abstractmethod
    def get_state_space(self):
        pass


class SawyerPlanningContext(AbstractPlanningContext):

    def __init__(self, state_space=None):
        self.state_space = SawyerConfigurationSpace() if state_space is None else state_space

    def get_state_space(self):
        return self.state_space

