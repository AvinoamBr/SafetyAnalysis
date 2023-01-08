from functools import partial

import pycrs
import pyproj


class CoordinateConvector(object):
    def __init__(self):
        ITM = pycrs.utils.search_name('israeli TM Grid')
        ITM = pycrs.parse.from_proj4(next(ITM)['proj4'])
        ITM_proj4 = ITM.to_proj4()

        WGS84 = pycrs.parse.from_epsg_code(4326) # WGS84 projection from epsg code
        WGS84_proj4 = WGS84.to_proj4()

        self.WGS84proj = pyproj.Proj(WGS84_proj4)
        self.ITMproj = pyproj.Proj(ITM_proj4)

        self._from_WGS84_to_ITM = partial(pyproj.transform, self.WGS84proj, self.ITMproj)
        self._from_ITM_to_WGS84 = partial(pyproj.transform, self.ITMproj, self.WGS84proj)


    def _test_(self, lng=35.2137, lat=31.7683):
        E,N = self._from_WGS84_to_ITM(lng,lat)
        print(f"lng {lng:.5f} lat {lat:.5f} E {E:.1f} N {N:.1f}")
        lng, lat = self._from_ITM_to_WGS84(E, N)
        print(f"lng {lng:.5f} lat {lat:.5f} E {E:.1f} N {N:.1f}")

    def convert_pgw_ITM_to_extent(self, pgw_fn, image):
        with open(pgw_fn, 'r') as f:
            scale_x = float(f.readline().strip())
            _ = f.readline()
            _ = f.readline()
            scale_y = float(f.readline().strip())
            E = float(f.readline().strip())
            N = float(f.readline().strip())
        left, top = self._from_ITM_to_WGS84(E, N)
        x_pixels = image.shape[1]
        y_pixels = image.shape[0]
        E = E + scale_x * x_pixels
        N = N + scale_y * y_pixels
        right, bottom =  self._from_ITM_to_WGS84(E, N)
        extent = (left,right,top,bottom)
        return extent


