"""
Routing Service Package — modality-aware query routing.
"""

from services.routing_service.modality_router import (
    route_query,
    route_heuristic,
    get_primary_modality,
)

__all__ = [
    "route_query",
    "route_heuristic",
    "get_primary_modality",
]
