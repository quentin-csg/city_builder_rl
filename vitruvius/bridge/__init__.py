"""Bridge WebSocket entre le moteur Python et le rendu Godot."""

from __future__ import annotations


def serve(*args, **kwargs):  # type: ignore[no-untyped-def]
    from vitruvius.bridge.server import serve as _serve
    return _serve(*args, **kwargs)


def main() -> None:
    from vitruvius.bridge.server import main as _main
    _main()


__all__ = ["serve", "main"]
