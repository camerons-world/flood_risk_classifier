"""Shared geo utilities. Distance calc was previously duplicated between
src/flood_warnings.py (lat/long haversine) and src/stations.py (easting/
northing euclidean) -- centralizing the lat/long version here since the new
postcode-lookup feature needs a third copy otherwise.
"""
import math


def haversine_km(lat1: float, long1: float, lat2: float, long2: float) -> float:
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(long2 - long1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
