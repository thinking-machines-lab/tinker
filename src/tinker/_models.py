from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, Optional, cast
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, TypeVar, Union, cast

import pydantic
from pydantic.fields import FieldInfo
from typing_extensions import (
    List,
    Unpack,
    Literal,
    ClassVar,
    ParamSpec,
    Protocol,
    Required,
    TypedDict,
    TypeGuard,
    Unpack,
    final,
    runtime_checkable,
)

from ._compat import (
    PYDANTIC_V2,
    ConfigDict,
    field_get_default,
    get_args,
    get_origin,
    is_literal_type,
    is_union,
    parse_obj,
)
from ._compat import (
    GenericModel as BaseGenericModel,
)
from ._constants import RAW_RESPONSE_HEADER
from ._types import (
    AnyMapping,
    Body,
    Headers,
    HttpxRequestFiles,
    NotGiven,
    Query,
    Timeout,
)
from ._utils import (
    PropertyInfo,
    extract_type_arg,
    is_annotated_type,
    is_given,
    is_list,
    is_mapping,
    is_type_alias_type,
    lru_cache,
    parse_date,
    parse_datetime,
    strip_annotated_type,
    strip_not_given,
)

if TYPE_CHECKING:
    from pydantic_core.core_schema import LiteralSchema, ModelField, ModelFieldsSchema, ModelSchema

__all__ = ["StrictBase", "BaseModel", "GenericModel"]

_T = TypeVar("_T")
_BaseModelT = TypeVar("_BaseModelT", bound=pydantic.BaseModel)

P = ParamSpec("P")


@runtime_checkable
class _ConfigProtocol(Protocol):
    allow_population_by_field_name: bool


class StrictBase(pydantic.BaseModel):
    """
    Don't allow extra fields, so user errors are caught earlier.
    Use this for request types.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)

class BaseModel(pydantic.BaseModel):
    """
    Use for classes that may appear in responses. Allow extra fields, so old clients can still work.
    """

    # For future-proofing, we ignore extra fields in case the server adds new fields.
    model_config = ConfigDict(frozen=True, extra="ignore")

    def __str__(self) -> str:
        return repr(self)


def _construct_field(value: object, field: FieldInfo, key: str) -> object:
    if value is None:
        return field_get_default(field)

    if PYDANTIC_V2:
        type_ = field.annotation
    else:
        type_ = cast(type, field.outer_type_)  # type: ignore

    if type_ is None:
        raise RuntimeError(f"Unexpected field type is None for {key}")

    return construct_type(value=value, type_=type_, metadata=getattr(field, "metadata", None))


def is_basemodel(type_: type) -> bool:
    """Returns whether or not the given type is either a `BaseModel` or a union of `BaseModel`"""
    if is_union(type_):
        for variant in get_args(type_):
            if is_basemodel(variant):
                return True

        return False

    return is_basemodel_type(type_)


def is_basemodel_type(type_: type) -> TypeGuard[type[BaseModel] | type[GenericModel]]:
    origin = get_origin(type_) or type_
    if not inspect.isclass(origin):
        return False
    return issubclass(origin, BaseModel) or issubclass(origin, GenericModel)


def build(
    base_model_cls: Callable[P, _BaseModelT],
    *args: P.args,
    **kwargs: P.kwargs,
) -> _BaseModelT:
    """Construct a BaseModel class without validation.

    This is useful for cases where you need to instantiate a `BaseModel`
    from an API response as this provides type-safe params which isn't supported
    by helpers like `construct_type()`.

    ```py
    build(MyModel, my_field_a="foo", my_field_b=123)
    ```
    """
    if args:
        raise TypeError(
            "Received positional arguments which are not supported; Keyword arguments must be used instead",
        )

    return cast(_BaseModelT, construct_type(type_=base_model_cls, value=kwargs))


def construct_type_unchecked(*, value: object, type_: type[_T]) -> _T:
    """Loose coercion to the expected type with construction of nested values.

    Note: the returned value from this function is not guaranteed to match the
    given type.
    """
    return cast(_T, construct_type(value=value, type_=type_))


def construct_type(*, value: object, type_: object, metadata: Optional[List[Any]] = None) -> object:
    """Loose coercion to the expected type with construction of nested values.

    If the given value does not match the expected type then it is returned as-is.
    """

    # store a reference to the original type we were given before we extract any inner
    # types so that we can properly resolve forward references in `TypeAliasType` annotations
    original_type = None

    # we allow `object` as the input type because otherwise, passing things like
    # `Literal['value']` will be reported as a type error by type checkers
    type_ = cast("type[object]", type_)
    if is_type_alias_type(type_):
        original_type = type_  # type: ignore[unreachable]
        type_ = type_.__value__  # type: ignore[unreachable]

    # unwrap `Annotated[T, ...]` -> `T`
    if metadata is not None and len(metadata) > 0:
        meta: tuple[Any, ...] = tuple(metadata)
    elif is_annotated_type(type_):
        meta = get_args(type_)[1:]
        type_ = extract_type_arg(type_, 0)
    else:
        meta = tuple()

    # we need to use the origin class for any types that are subscripted generics
    # e.g. Dict[str, object]
    origin = get_origin(type_) or type_
    args = get_args(type_)

    if is_union(origin):
        try:
            return validate_type(type_=cast("type[object]", original_type or type_), value=value)
        except Exception:
            pass

        # if the type is a discriminated union then we want to construct the right variant
        # in the union, even if the data doesn't match exactly, otherwise we'd break code
        # that relies on the constructed class types, e.g.
        #
        # class FooType:
        #   kind: Literal['foo']
        #   value: str
        #
        # class BarType:
        #   kind: Literal['bar']
        #   value: int
        #
        # without this block, if the data we get is something like `{'kind': 'bar', 'value': 'foo'}` then
        # we'd end up constructing `FooType` when it should be `BarType`.
        discriminator = _build_discriminated_union_meta(union=type_, meta_annotations=meta)
        if discriminator and is_mapping(value):
            variant_value = value.get(discriminator.field_alias_from or discriminator.field_name)
            if variant_value and isinstance(variant_value, str):
                variant_type = discriminator.mapping.get(variant_value)
                if variant_type:
                    return construct_type(type_=variant_type, value=value)

        # if the data is not valid, use the first variant that doesn't fail while deserializing
        for variant in args:
            try:
                return construct_type(value=value, type_=variant)
            except Exception:
                continue

        raise RuntimeError(f"Could not convert data into a valid instance of {type_}")

    if origin == dict:
        if not is_mapping(value):
            return value

        _, items_type = get_args(type_)  # Dict[_, items_type]
        return {key: construct_type(value=item, type_=items_type) for key, item in value.items()}

    if (
        not is_literal_type(type_)
        and inspect.isclass(origin)
        and (issubclass(origin, BaseModel) or issubclass(origin, GenericModel))
    ):
        if is_list(value):
            return [
                cast(Any, type_).construct(**entry) if is_mapping(entry) else entry
                for entry in value
            ]

        if is_mapping(value):
            if issubclass(type_, BaseModel):
                return type_.construct(**value)  # type: ignore[arg-type]

            return cast(Any, type_).construct(**value)

    if origin == list:
        if not is_list(value):
            return value

        inner_type = args[0]  # List[inner_type]
        return [construct_type(value=entry, type_=inner_type) for entry in value]

    if origin == float:
        if isinstance(value, int):
            coerced = float(value)
            if coerced != value:
                return value
            return coerced

        return value

    if type_ == datetime:
        try:
            return parse_datetime(value)  # type: ignore
        except Exception:
            return value

    if type_ == date:
        try:
            return parse_date(value)  # type: ignore
        except Exception:
            return value

    return value


@runtime_checkable
class CachedDiscriminatorType(Protocol):
    __discriminator__: DiscriminatorDetails


class DiscriminatorDetails:
    field_name: str
    """The name of the discriminator field in the variant class, e.g.

    ```py
    class Foo(BaseModel):
        type: Literal['foo']
    ```

    Will result in field_name='type'
    """

    field_alias_from: str | None
    """The name of the discriminator field in the API response, e.g.

    ```py
    class Foo(BaseModel):
        type: Literal['foo'] = Field(alias='type_from_api')
    ```

    Will result in field_alias_from='type_from_api'
    """

    mapping: dict[str, type]
    """Mapping of discriminator value to variant type, e.g.

    {'foo': FooVariant, 'bar': BarVariant}
    """

    def __init__(
        self,
        *,
        mapping: dict[str, type],
        discriminator_field: str,
        discriminator_alias: str | None,
    ) -> None:
        self.mapping = mapping
        self.field_name = discriminator_field
        self.field_alias_from = discriminator_alias


def _build_discriminated_union_meta(
    *, union: type, meta_annotations: tuple[Any, ...]
) -> DiscriminatorDetails | None:
    if isinstance(union, CachedDiscriminatorType):
        return union.__discriminator__

    discriminator_field_name: str | None = None

    for annotation in meta_annotations:
        if isinstance(annotation, PropertyInfo) and annotation.discriminator is not None:
            discriminator_field_name = annotation.discriminator
            break

    if not discriminator_field_name:
        return None

    mapping: dict[str, type] = {}
    discriminator_alias: str | None = None

    for variant in get_args(union):
        variant = strip_annotated_type(variant)
        if is_basemodel_type(variant):
            if PYDANTIC_V2:
                field = _extract_field_schema_pv2(variant, discriminator_field_name)
                if not field:
                    continue

                # Note: if one variant defines an alias then they all should
                discriminator_alias = field.get("serialization_alias")

                field_schema = field["schema"]

                if field_schema["type"] == "literal":
                    for entry in cast("LiteralSchema", field_schema)["expected"]:
                        if isinstance(entry, str):
                            mapping[entry] = variant
            else:
                field_info = cast("dict[str, FieldInfo]", variant.__fields__).get(
                    discriminator_field_name
                )  # pyright: ignore[reportDeprecated, reportUnnecessaryCast]
                if not field_info:
                    continue

                # Note: if one variant defines an alias then they all should
                discriminator_alias = field_info.alias

                if (annotation := getattr(field_info, "annotation", None)) and is_literal_type(
                    annotation
                ):
                    for entry in get_args(annotation):
                        if isinstance(entry, str):
                            mapping[entry] = variant

    if not mapping:
        return None

    details = DiscriminatorDetails(
        mapping=mapping,
        discriminator_field=discriminator_field_name,
        discriminator_alias=discriminator_alias,
    )
    cast(CachedDiscriminatorType, union).__discriminator__ = details
    return details


def _extract_field_schema_pv2(model: type[BaseModel], field_name: str) -> ModelField | None:
    schema = model.__pydantic_core_schema__
    if schema["type"] == "definitions":
        schema = schema["schema"]

    if schema["type"] != "model":
        return None

    schema = cast("ModelSchema", schema)
    fields_schema = schema["schema"]
    if fields_schema["type"] != "model-fields":
        return None

    fields_schema = cast("ModelFieldsSchema", fields_schema)
    field = fields_schema["fields"].get(field_name)
    if not field:
        return None

    return cast("ModelField", field)  # pyright: ignore[reportUnnecessaryCast]


def validate_type(*, type_: type[_T], value: object) -> _T:
    """Strict validation that the given value matches the expected type"""
    if inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel):
        return cast(_T, parse_obj(type_, value))

    return cast(_T, _validate_non_model_type(type_=type_, value=value))


def set_pydantic_config(typ: Any, config: pydantic.ConfigDict) -> None:
    """Add a pydantic config for the given type.

    Note: this is a no-op on Pydantic v1.
    """
    setattr(typ, "__pydantic_config__", config)  # noqa: B010


# our use of subclassing here causes weirdness for type checkers,
# so we just pretend that we don't subclass
if TYPE_CHECKING:
    GenericModel = BaseModel
else:

    class GenericModel(BaseGenericModel, BaseModel):
        pass


if PYDANTIC_V2:
    from pydantic import TypeAdapter as _TypeAdapter

    _CachedTypeAdapter = cast("TypeAdapter[object]", lru_cache(maxsize=None)(_TypeAdapter))

    if TYPE_CHECKING:
        from pydantic import TypeAdapter
    else:
        TypeAdapter = _CachedTypeAdapter

    def _validate_non_model_type(*, type_: type[_T], value: object) -> _T:
        return TypeAdapter(type_).validate_python(value)

elif not TYPE_CHECKING:  # TODO: condition is weird

    class RootModel(GenericModel, Generic[_T]):
        """Used as a placeholder to easily convert runtime types to a Pydantic format
        to provide validation.

        For example:
        ```py
        validated = RootModel[int](__root__="5").__root__
        # validated: 5
        ```
        """

        __root__: _T

    def _validate_non_model_type(*, type_: type[_T], value: object) -> _T:
        model = _create_pydantic_model(type_).validate(value)
        return cast(_T, model.__root__)

    def _create_pydantic_model(type_: _T) -> Type[RootModel[_T]]:
        return RootModel[type_]  # type: ignore


class FinalRequestOptionsInput(TypedDict, total=False):
    method: Required[str]
    url: Required[str]
    params: Query
    headers: Headers
    max_retries: int
    timeout: float | Timeout | None
    files: HttpxRequestFiles | None
    idempotency_key: str
    json_data: Body
    extra_json: AnyMapping
    follow_redirects: bool


@final
class FinalRequestOptions(pydantic.BaseModel):
    method: str
    url: str
    params: Query = {}
    headers: Union[Headers, NotGiven] = NotGiven()
    max_retries: Union[int, NotGiven] = NotGiven()
    timeout: Union[float, Timeout, None, NotGiven] = NotGiven()
    files: Union[HttpxRequestFiles, None] = None
    idempotency_key: Union[str, None] = None
    post_parser: Union[Callable[[Any], Any], NotGiven] = NotGiven()
    follow_redirects: Union[bool, None] = None

    # It should be noted that we cannot use `json` here as that would override
    # a BaseModel method in an incompatible fashion.
    json_data: Union[Body, None] = None
    extra_json: Union[AnyMapping, None] = None

    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config(pydantic.BaseConfig):  # pyright: ignore[reportDeprecated]
            arbitrary_types_allowed: bool = True

    def get_max_retries(self, max_retries: int) -> int:
        if isinstance(self.max_retries, NotGiven):
            return max_retries
        return self.max_retries

    def _strip_raw_response_header(self) -> None:
        if not is_given(self.headers):
            return

        if self.headers.get(RAW_RESPONSE_HEADER):
            self.headers = {**self.headers}
            self.headers.pop(RAW_RESPONSE_HEADER)

    # override the `construct` method so that we can run custom transformations.
    # this is necessary as we don't want to do any actual runtime type checking
    # (which means we can't use validators) but we do want to ensure that `NotGiven`
    # values are not present
    #
    # type ignore required because we're adding explicit types to `**values`
    @classmethod
    def construct(  # type: ignore
        cls,
        _fields_set: set[str] | None = None,
        **values: Unpack[FinalRequestOptionsInput],
    ) -> FinalRequestOptions:
        kwargs: dict[str, Any] = {
            # we unconditionally call `strip_not_given` on any value
            # as it will just ignore any non-mapping types
            key: strip_not_given(value)
            for key, value in values.items()
        }
        if PYDANTIC_V2:
            return super().model_construct(_fields_set, **kwargs)
        return cast(FinalRequestOptions, super().construct(_fields_set, **kwargs))  # pyright: ignore[reportDeprecated]

    if not TYPE_CHECKING:
        # type checkers incorrectly complain about this assignment
        model_construct = construct
