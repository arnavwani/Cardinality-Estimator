from abc import ABC, abstractmethod


class Estimator(ABC):
    name = "base"
    #Return an estimated cardinality (row count) for the given Query.
    @abstractmethod
    def estimate(self, query) -> float:
        raise NotImplementedError