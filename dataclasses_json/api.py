import abc
import json
from enum import Enum
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union, overload)

from dataclasses_json.stringcase import (camelcase, pascalcase, snakecase,
                                         spinalcase)  # type: ignore
from dataclasses_json.cfg import config
from dataclasses_json.core import (Json, _ExtendedEncoder, _asdict,
                                   _decode_dataclass,
                                   _register_generic_cls, _register_optional_cls)
from dataclasses_json.mm import (JsonData, SchemaType, build_schema)
from dataclasses_json.undefined import Undefined
from dataclasses_json.utils import (_handle_undefined_parameters_safe,
                                    _undefined_parameter_action_safe)

A = TypeVar('A', bound="DataClassJsonMixin")
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D', bound="DecodableGenericABC")
Fields = List[Tuple[str, Any]]


class LetterCase(Enum):
    CAMEL = camelcase
    KEBAB = spinalcase
    SNAKE = snakecase
    PASCAL = pascalcase


class DataClassJsonMixin(abc.ABC):
    """
    DataClassJsonMixin is an ABC that functions as a Mixin.

    As with other ABCs, it should not be instantiated directly.
    """
    dataclass_json_config = None

    def to_json(self,
                *,
                skipkeys: bool = False,
                ensure_ascii: bool = True,
                check_circular: bool = True,
                allow_nan: bool = True,
                indent: Optional[Union[int, str]] = None,
                separators: Tuple[str, str] = None,
                default: Callable = None,
                sort_keys: bool = False,
                **kw) -> str:
        return json.dumps(self.to_dict(encode_json=False),
                          cls=_ExtendedEncoder,
                          skipkeys=skipkeys,
                          ensure_ascii=ensure_ascii,
                          check_circular=check_circular,
                          allow_nan=allow_nan,
                          indent=indent,
                          separators=separators,
                          default=default,
                          sort_keys=sort_keys,
                          **kw)

    @classmethod
    def from_json(cls: Type[A],
                  s: JsonData,
                  *,
                  parse_float=None,
                  parse_int=None,
                  parse_constant=None,
                  infer_missing=False,
                  **kw) -> A:
        kvs = json.loads(s,
                         parse_float=parse_float,
                         parse_int=parse_int,
                         parse_constant=parse_constant,
                         **kw)
        return cls.from_dict(kvs, infer_missing=infer_missing)

    @classmethod
    def from_dict(cls: Type[A],
                  kvs: Json,
                  *,
                  infer_missing=False) -> A:
        return _decode_dataclass(cls, kvs, infer_missing)

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)

    @classmethod
    def schema(cls: Type[A],
               *,
               infer_missing: bool = False,
               only=None,
               exclude=(),
               many: bool = False,
               context=None,
               load_only=(),
               dump_only=(),
               partial: bool = False,
               unknown=None) -> SchemaType:
        Schema = build_schema(cls, DataClassJsonMixin, infer_missing, partial)

        if unknown is None:
            undefined_parameter_action = _undefined_parameter_action_safe(cls)
            if undefined_parameter_action is not None:
                # We can just make use of the same-named mm keywords
                unknown = undefined_parameter_action.name.lower()

        return Schema(only=only,
                      exclude=exclude,
                      many=many,
                      context=context,
                      load_only=load_only,
                      dump_only=dump_only,
                      partial=partial,
                      unknown=unknown)


class DecodableGenericABC(abc.ABC):
    """
    DecodableGenericMixin is an ABC that supports supports
    user-defined generic classes encoding and decoding.

    Child classes MUST define __encode__ and __decode__ methods.

    An optional class parameter `num_args` defines the number
    of type parameters, or 1 if not present.
    (So you should use `num_args=2` for dict-like classes, as example.)

    As with other ABCs, it should not be instantiated directly.
    """

    __slots__ = ('_num_args')
    _num_args: int

    def __init_subclass__(cls, *, num_args: int = 1, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._num_args = num_args
        _register_generic_cls(cls, cls.encode, cls.decode.__func__)

    @abc.abstractmethod
    def __encode__(self, *, data_encoder: Callable[[B], Json], **kwargs) -> Json:
        """
        An instance encoder implementation.
        Should not be worried for the number of type parameters,
        it is checked on the interface level.

        Args:
            data_encoder: A library-provided function that encodes
                          the given data value into an appropriate JSON-like data.
            **kwargs: Any arbitrary parameters passed to the encoder.
                      Reserved for future uses.
        """
        
        raise NotImplementedError

    @overload
    def encode(self, *, data_encoder: Callable[[B], Json], **kwargs) -> Json:
        """ Encodes the given class instance into a JSON-like dict. """
        pass

    def encode(self, *args, **kwargs):
        """ An interfaces for encoding class instances. """
        return self.__encode__(*args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def __decode__(cls: Type[D],
                   data: Json,
                   *types: Type[B],
                   data_decoder: Callable[[Type[B], Json], B],
                   **kwargs) -> D:
        """
        A class decoder implementation.
        Should not be worried for the number of type parameters,
        it is checked on the interface level.

        Args:
            data: A JSON to decode.
            *types: A list of type parameters values for the class.
            data_decoder: A library-provided function that decodes
                          the given data type from the given encoded value
            **kwargs: Any arbitrary parameters passed to the decoder.
                      Reserved for future uses.
        """

        raise NotImplementedError

    @classmethod
    @overload
    def decode(cls: Type[D], data: Json, *types: Type[B], data_decoder: Callable[[Type[B], Json], B], **kwargs) -> D:
        """ Decodes class from the given JSON-like dict `data`. """
        pass

    @classmethod
    def decode(cls, data: Json, *types: Type[B], **kwargs):
        """ An interfaces for decoding class. """
        if (len(types) != cls._num_args):
            raise TypeError(f"{'Too many' if len(types) > cls._num_args else 'Not enough'} types for decoding {cls.__name__}: Expected {cls._num_args}, got {len(types)}.")

        return cls.__decode__(data, *types, **kwargs)

class OptionalABC(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super(OptionalABC, cls).__init_subclass__(**kwargs)
        _register_optional_cls(cls, cls.__empty__)
    
    @classmethod
    @abc.abstractmethod
    def __empty__(cls):
        raise NotImplementedError


def dataclass_json(_cls=None, *, letter_case=None,
                   undefined: Optional[Union[str, Undefined]] = None):
    """
    Based on the code in the `dataclasses` module to handle optional-parens
    decorators. See example below:

    @dataclass_json
    @dataclass_json(letter_case=LetterCase.CAMEL)
    class Example:
        ...
    """

    def wrap(cls):
        return _process_class(cls, letter_case, undefined)

    if _cls is None:
        return wrap
    return wrap(_cls)


def _process_class(cls, letter_case, undefined):
    if letter_case is not None or undefined is not None:
        cls.dataclass_json_config = config(letter_case=letter_case,
                                           undefined=undefined)[
            'dataclasses_json']

    cls.to_json = DataClassJsonMixin.to_json
    # unwrap and rewrap classmethod to tag it to cls rather than the literal
    # DataClassJsonMixin ABC
    cls.from_json = classmethod(DataClassJsonMixin.from_json.__func__)
    cls.to_dict = DataClassJsonMixin.to_dict
    cls.from_dict = classmethod(DataClassJsonMixin.from_dict.__func__)
    cls.schema = classmethod(DataClassJsonMixin.schema.__func__)

    cls.__init__ = _handle_undefined_parameters_safe(cls, kvs=(), usage="init")
    # register cls as a virtual subclass of DataClassJsonMixin
    DataClassJsonMixin.register(cls)
    return cls
