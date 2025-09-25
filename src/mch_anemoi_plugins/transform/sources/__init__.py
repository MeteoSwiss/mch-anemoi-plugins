try:
    from .weathermart import DataProviderSource, DHM25, OPERA, NASADEM, SATELLITE, SURFACE
except ImportError as e:
    raise ImportError("Could not import weathermart-based sources. Make sure weathermart is installed.") from e

__all__ = [
    "DataProviderSource",
    "DHM25",
    "OPERA",
    "NASADEM",
    "SATELLITE",
    "SURFACE",
]