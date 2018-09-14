"""W7-X MDSplus acquisition module."""
from pyfusion.acquisition.base import BaseAcquisition


class W7MAcquisition(BaseAcquisition):
    """Acquisition class for W7X MDSplus data system.

    """
    def __init__(self, *args, **kwargs):
        super(W7MAcquisition, self).__init__(*args, **kwargs)
