from .client import FederatedClient
from .server import FederatedServer
from .aggregation import aggregate_updates, compute_update_geometry
from .losses import InfoNCELoss, AlignmentLoss, AnchoringLoss
from .fedprox import FedProxClient
from .fedper import FedPerClient
from .moon import MOONClient

# Additional baseline methods
from .MFCPL.mfcpl_client import MFCPLClient
from .MFCPL.mfcpl_server import MFCPLServer
from .MFCPL.mfcpl_losses import MFCPLCombinedLoss, CMPRLoss, CMPCLoss, CMALoss

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'FedProxClient',
    'FedPerClient',
    'MOONClient',
    'MFCPLClient',
    'MFCPLServer',
    'MFCPLCombinedLoss',
    'CMPRLoss',
    'CMPCLoss',
    'CMALoss',
    'aggregate_updates',
    'compute_update_geometry',
    'InfoNCELoss',
    'AlignmentLoss',
    'AnchoringLoss',
]
