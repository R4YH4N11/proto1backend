"""Entrypoint shim so `uvicorn main:app` works from the project root."""

from app.main import app

__all__ = ["app"]
