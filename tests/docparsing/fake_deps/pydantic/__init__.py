from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, TypeVar

__all__ = [
    "BaseModel",
    "ConfigDict",
    "Field",
    "GetCoreSchemaHandler",
    "GetJsonSchemaHandler",
    "PrivateAttr",
    "ValidationError",
    "field_validator",
    "model_validator",
]

_FIELD_UNSET = object()
T = TypeVar("T")


class _FieldInfo:
    def __init__(
        self, *, default: Any = _FIELD_UNSET, default_factory: Callable[[], Any] | None = None
    ) -> None:
        self.default = default
        self.default_factory = default_factory


def Field(
    *args: Any,
    default: Any = _FIELD_UNSET,
    default_factory: Callable[[], Any] | None = None,
    **_kwargs: Any,
) -> _FieldInfo:
    if args:
        default = args[0]
    return _FieldInfo(default=default, default_factory=default_factory)


def ConfigDict(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


class GetCoreSchemaHandler:
    """Minimal shim returning the provided schema unchanged."""

    def __call__(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
        return schema


class GetJsonSchemaHandler:
    """Minimal shim compatible with pydantic's handler protocol."""

    def __call__(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
        return schema


def field_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return func

    return decorator


def model_validator(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return func

    return decorator


class _PydanticMeta(type):
    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> type:
        field_defaults: Dict[str, Any] = {}
        for key, value in list(namespace.items()):
            if isinstance(value, _FieldInfo):
                if value.default_factory is not None:
                    default_value = value.default_factory()
                elif value.default is not _FIELD_UNSET:
                    default_value = value.default
                else:
                    default_value = None
                namespace[key] = default_value
                field_defaults[key] = default_value
        namespace.setdefault("__fields_defaults__", {})
        namespace["__fields_defaults__"].update(field_defaults)
        return super().__new__(mcls, name, bases, namespace)


class BaseModel(metaclass=_PydanticMeta):
    model_config: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        data: Dict[str, Any] = dict(getattr(self, "__fields_defaults__", {}))
        data.update(kwargs)
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def model_rebuild(cls) -> None:  # pragma: no cover - compatibility method
        return None

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "model_dump"):
                result[key] = value.model_dump(*args, **kwargs)
            elif isinstance(value, list):
                result[key] = [
                    item.model_dump(*args, **kwargs) if hasattr(item, "model_dump") else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class ValidationError(Exception):
    """Compatibility shim for pydantic.ValidationError."""


def PrivateAttr(default: Any = None) -> Any:
    """Return the supplied default for compatibility with pydantic."""

    return default
