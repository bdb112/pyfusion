"""W7-X acquisition module."""
from pyfusion.acquisition.base import BaseAcquisition


class W7XAcquisition(BaseAcquisition):
    """Acquisition class for W7X data system.

    """
    def __init__(self, *args, **kwargs):
        super(W7XAcquisition, self).__init__(*args, **kwargs)

