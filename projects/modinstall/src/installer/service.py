from typing import Any, Callable

# Static channel registry mapping channel names to handler callables
_CHANNEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_channel(channel_name: str, handler: Callable[..., Any]) -> None:
    """Register a handler callable for a specific channel name."""
    _CHANNEL_REGISTRY[channel_name] = handler


def dispatch(channel_name: str, payload: dict[str, Any]) -> Any:
    """Dispatch a payload to the registered handler for the given channel name.

    Raises KeyError if the channel is unregistered.
    """
    if channel_name not in _CHANNEL_REGISTRY:
        raise KeyError(f"Channel '{channel_name}' is unregistered")
    
    handler = _CHANNEL_REGISTRY[channel_name]
    return handler(payload)


class Service:
    """Service pattern class supporting Service.method routing conventions."""

    @staticmethod
    def method(channel_name: str, payload: dict[str, Any]) -> Any:
        """Route a service method call via the dispatch mechanism."""
        return dispatch(channel_name, payload)
