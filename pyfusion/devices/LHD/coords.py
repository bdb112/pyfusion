"""
"""
from pyfusion.data.base import BaseCoordTransform

class LHDConvenienceCoordTransform(BaseCoordTransform):
    input_coords = 'cylindricaal'
    output_coords = 'toroidal'

    def transform(self, coords, kh=None):
        mag_angle = map_kappa_h_mag_angle(coords, kh)
        return (mag_angle, 0, 0)
