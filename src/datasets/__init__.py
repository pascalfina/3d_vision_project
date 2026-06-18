from .scan3r_obj import Scan3RObjectDataset
from .scan3r_obj_obj import Scan3RObjObjDataset
from .scan3r_obj_patch import Scan3RPatchObjectDataset
from .scan3r_scene import Scan3RSceneGraphDataset
from .scannet_obj import ScannetObjectDataset

# Optional sparse-view completion dataset. Some inference paths only need the
# scene/object datasets, so missing step2 sources should not break SLAT/U3DGS.
try:
    from .step2_slat_pairs import Step2SLATPairDataset
except ModuleNotFoundError:  # pragma: no cover - optional dataset
    Step2SLATPairDataset = None
