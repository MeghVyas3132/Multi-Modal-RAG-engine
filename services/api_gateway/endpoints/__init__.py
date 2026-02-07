"""
API Gateway endpoint routers — mounted by the main app.
"""

from services.api_gateway.endpoints.search import router as search_router
from services.api_gateway.endpoints.chat import router as chat_router
from services.api_gateway.endpoints.upload import router as upload_router
from services.api_gateway.endpoints.web import router as web_router
from services.api_gateway.endpoints.graph import router as graph_router
from services.api_gateway.endpoints.health import router as health_router

__all__ = [
    "search_router",
    "chat_router",
    "upload_router",
    "web_router",
    "graph_router",
    "health_router",
]
