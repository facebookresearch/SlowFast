from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets.kinetics import Kinetics

@DATASET_REGISTRY.register()
class Custom(Kinetics):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        # You can override methods here if needed
