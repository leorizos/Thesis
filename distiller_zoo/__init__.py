from .FitNet import HintLoss
from .AT import Attention
from .KD import DistillKL
from .SP import Similarity
from .VID import VIDLoss
from .SemCKD import SemCKDLoss
from .PKT import PKT
from .softPKT2 import SoftPKT2
from .AKD import akd_loss, AnchorNet, calculate_anchor_set
from .SoftAKD import soft_akd_loss, GCN, soften_sigma_with_gcn
from .SoftAKD2 import monitor_gcn