# Dataclasses JSON

![](https://github.com/lidatong/dataclasses-json/workflows/dataclasses-json/badge.svg)

This library provides a simple API for encoding and decoding [dataclasses](https://docs.python.org/3/library/dataclasses.html) to and from JSON.

It's very easy to get started.

[README / Documentation website](https://lidatong.github.io/dataclasses-json). Features a navigation bar and search functionality, and should mirror this README exactly -- take a look!

## Quickstart

`pip install dataclasses-json`

```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Person:
    name: str


person = Person(name='lidatong')
person.to_json()  # '{"name": "lidatong"}' <- this is a string
person.to_dict()  # {'name': 'lidatong'} <- this is a dict
Person.from_json('{"name": "lidatong"}')  # Person(1)
Person.from_dict({'name': 'lidatong'})  # Person(1)

# You can also apply _schema validation_ using an alternative API
# This can be useful for "typed" Python code

Person.from_json('{"name": 42}')  # This is ok. 42 is not a `str`, but
                                  # dataclass creation does not validate types
Person.schema().loads('{"name": 42}')  # Error! Raises `ValidationError`
```

**What if you want to work with camelCase JSON?**

```python
# same imports as above, with the additional `LetterCase` import
from dataclasses import dataclass
from dataclasses_json import dataclass_json, LetterCase

@dataclass_json(letter_case=LetterCase.CAMEL)  # now all fields are encoded/decoded from camelCase
@dataclass
class ConfiguredSimpleExample:
    int_field: int

ConfiguredSimpleExample(1).to_json()  # {"intField": 1}
ConfiguredSimpleExample.from_json('{"intField": 1}')  # ConfiguredSimpleExample(1)
```

## Supported types

It's recursive (see caveats below), so you can easily work with nested dataclasses.
In addition to the supported types in the 
[py to JSON table](https://docs.python.org/3/library/json.html#py-to-json-table), this library supports the following:

- any arbitrary [Collection](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection) type is supported.
[Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) types are encoded as JSON objects and `str` types as JSON strings. 
Any other Collection types are encoded into JSON arrays, but decoded into the original collection types.

- [datetime](https://docs.python.org/3/library/datetime.html#available-types) 
objects. `datetime` objects are encoded to `float` (JSON number) using 
[timestamp](https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp).
As specified in the `datetime` docs, if your `datetime` object is naive, it will 
assume your system local timezone when calling `.timestamp()`. JSON numbers 
corresponding to a `datetime` field in your dataclass are decoded 
into a datetime-aware object, with `tzinfo` set to your system local timezone.
Thus, if you encode a datetime-naive object, you will decode into a 
datetime-aware object. This is important, because encoding and decoding won't 
strictly be inverses. See [this section](#Overriding) if you want to override this default
behavior (for example, if you want to use ISO).

- [UUID](https://docs.python.org/3/library/uuid.html#uuid.UUID) objects. They 
are encoded as `str` (JSON string).

- [Decimal](https://docs.python.org/3/library/decimal.html) objects. They are
also encoded as `str`.

**The [latest release](https://github.com/lidatong/dataclasses-json/releases/latest) is compatible with both Python 3.7 and Python 3.6 (with the dataclasses backport).**

## Usage

#### Approach 1: Class decorator

```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Person:
    name: str

lidatong = Person('lidatong')

# Encoding to JSON
lidatong.to_json()  # '{"name": "lidatong"}'

# Decoding from JSON
Person.from_json('{"name": "lidatong"}')  # Person(name='lidatong')
```

Note that the `@dataclass_json` decorator must be stacked above the `@dataclass`
decorator (order matters!)

#### Approach 2: Inherit from a mixin

```python
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

@dataclass
class Person(DataClassJsonMixin):
    name: str

lidatong = Person('lidatong')

# A different example from Approach 1 above, but usage is the exact same
assert Person.from_json(lidatong.to_json()) == lidatong
```

Pick whichever approach suits your taste. Note that there is better support for
 the mixin approach when using _static analysis_ tools (e.g. linting, typing),
 but the differences in implementation will be invisible in _runtime_ usage.

## How do I...



### Use my dataclass with JSON arrays or objects?

```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Person:
    name: str
```

**Encode into a JSON array containing instances of my Data Class**

```python
people_json = [Person('lidatong')]
Person.schema().dumps(people_json, many=True)  # '[{"name": "lidatong"}]'
```

**Decode a JSON array containing instances of my Data Class**

```python
people_json = '[{"name": "lidatong"}]'
Person.schema().loads(people_json, many=True)  # [Person(name='lidatong')]
```

**Encode as part of a larger JSON object containing my Data Class (e.g. an HTTP 
request/response)**

```python
import json

response_dict = {
    'response': {
        'person': Person('lidatong').to_dict()
    }
}

response_json = json.dumps(response_dict)
```

In this case, we do two steps. First, we encode the dataclass into a 
**python dictionary** rather than a JSON string, using `.to_dict`. 

Second, we leverage the built-in `json.dumps` to serialize our `dataclass` into 
a JSON string.

**Decode as part of a larger JSON object containing my Data Class (e.g. an HTTP 
response)**

```python
import json

response_dict = json.loads('{"response": {"person": {"name": "lidatong"}}}')

person_dict = response_dict['response']

person = Person.from_dict(person_dict)
```

In a similar vein to encoding above, we leverage the built-in `json` module.

First, call `json.loads` to read the entire JSON object into a 
dictionary. We then access the key of the value containing the encoded dict of 
our `Person` that we want to decode (`response_dict['response']`).

Second, we load in the dictionary using `Person.from_dict`.


### Encode or decode into Python lists/dictionaries rather than JSON?

This can be by calling `.schema()` and then using the corresponding 
encoder/decoder methods, ie. `.load(...)`/`.dump(...)`.

**Encode into a single Python dictionary**

```python
person = Person('lidatong')
person.to_dict()  # {'name': 'lidatong'}
```

**Encode into a list of Python dictionaries**

```python
people = [Person('lidatong')]
Person.schema().dump(people, many=True)  # [{'name': 'lidatong'}]
```

**Decode a dictionary into a single dataclass instance**

```python
person_dict = {'name': 'lidatong'}
Person.from_dict(person_dict)  # Person(name='lidatong')
```

**Decode a list of dictionaries into a list of dataclass instances**

```python
people_dicts = [{"name": "lidatong"}]
Person.schema().load(people_dicts, many=True)  # [Person(name='lidatong')]
```

### Encode or decode from camelCase (or kebab-case)?

JSON letter case by convention is camelCase, in Python members are by convention snake_case.

You can configure it to encode/decode from other casing schemes at both the class level and the field level.

```python
from dataclasses import dataclass, field

from dataclasses_json import LetterCase, config, dataclass_json


# changing casing at the class level
@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Person:
    given_name: str
    family_name: str
    
Person('Alice', 'Liddell').to_json()  # '{"givenName": "Alice"}'
Person.from_json('{"givenName": "Alice", "familyName": "Liddell"}')  # Person('Alice', 'Liddell')

# at the field level
@dataclass_json
@dataclass
class Person:
    given_name: str = field(metadata=config(letter_case=LetterCase.CAMEL))
    family_name: str
    
Person('Alice', 'Liddell').to_json()  # '{"givenName": "Alice"}'
# notice how the `family_name` field is still snake_case, because it wasn't configured above
Person.from_json('{"givenName": "Alice", "family_name": "Liddell"}')  # Person('Alice', 'Liddell')
```

**This library assumes your field follows the Python convention of snake_case naming.**
If your field is not `snake_case` to begin with and you attempt to parameterize `LetterCase`, 
the behavior of encoding/decoding is undefined (most likely it will result in subtle bugs).

### Encode or decode using a different name

```python
from dataclasses import dataclass, field

from dataclasses_json import config, dataclass_json

@dataclass_json
@dataclass
class Person:
    given_name: str = field(metadata=config(field_name="overriddenGivenName"))

Person(given_name="Alice")  # Person('Alice')
Person.from_json('{"overriddenGivenName": "Alice"}')  # Person('Alice')
Person('Alice').to_json()  # {"overriddenGivenName": "Alice"}
```

### Handle missing or optional field values when decoding?

By default, any fields in your dataclass that use `default` or 
`default_factory` will have the values filled with the provided default, if the
corresponding field is missing from the JSON you're decoding.

**Decode JSON with missing field**

```python
@dataclass_json
@dataclass
class Student:
    id: int
    name: str = 'student'

Student.from_json('{"id": 1}')  # Student(id=1, name='student')
```

Notice `from_json` filled the field `name` with the specified default 'student'
when it was missing from the JSON.

Sometimes you have fields that are typed as `Optional`, but you don't 
necessarily want to assign a default. In that case, you can use the 
`infer_missing` kwarg to make `from_json` infer the missing field value as `None`.

**Decode optional field without default**

```python
@dataclass_json
@dataclass
class Tutor:
    id: int
    student: Optional[Student] = None

Tutor.from_json('{"id": 1}')  # Tutor(id=1, student=None)
```

Personally I recommend you leverage dataclass defaults rather than using 
`infer_missing`, but if for some reason you need to decouple the behavior of 
JSON decoding from the field's default value, this will allow you to do so.


### Handle unknown / extraneous fields in JSON?

By default, it is up to the implementation what happens when a `json_dataclass` receives input parameters that are not defined.
(the `from_dict` method ignores them, when loading using `schema()` a ValidationError is raised.)
There are three ways to customize this behavior.

Assume you want to instantiate a dataclass with the following dictionary:
```python
dump_dict = {"endpoint": "some_api_endpoint", "data": {"foo": 1, "bar": "2"}, "undefined_field_name": [1, 2, 3]}
```

1. You can enforce to always raise an error by setting the `undefined` keyword to `Undefined.RAISE`
 (`'RAISE'` as a case-insensitive string works as well). Of course it works normally if you don't pass any undefined parameters.
    
```python
from dataclasses_json import Undefined

@dataclass_json(undefined=Undefined.RAISE)
@dataclass()
class ExactAPIDump:
    endpoint: str
    data: Dict[str, Any]

dump = ExactAPIDump.from_dict(dump_dict)  # raises UndefinedParameterError
```

2. You can simply ignore any undefined parameters by setting the `undefined` keyword to `Undefined.EXCLUDE`
 (`'EXCLUDE'` as a case-insensitive string works as well). Note that you will not be able to retrieve them using `to_dict`:
    
```python
from dataclasses_json import Undefined

@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass()
class DontCareAPIDump:
    endpoint: str
    data: Dict[str, Any]

dump = DontCareAPIDump.from_dict(dump_dict)  # DontCareAPIDump(endpoint='some_api_endpoint', data={'foo': 1, 'bar': '2'})
dump.to_dict()  # {"endpoint": "some_api_endpoint", "data": {"foo": 1, "bar": "2"}}
```

3. You can save them in a catch-all field and do whatever needs to be done later. Simply set the `undefined`
keyword to `Undefined.INCLUDE` (`'INCLUDE'` as a case-insensitive string works as well) and define a field
of type `CatchAll` where all unknown values will end up.
 This simply represents a dictionary that can hold anything. 
 If there are no undefined parameters, this will be an empty dictionary.
    
```python
from dataclasses_json import Undefined, CatchAll

@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass()
class UnknownAPIDump:
    endpoint: str
    data: Dict[str, Any]
    unknown_things: CatchAll

dump = UnknownAPIDump.from_dict(dump_dict)  # UnknownAPIDump(endpoint='some_api_endpoint', data={'foo': 1, 'bar': '2'}, unknown_things={'undefined_field_name': [1, 2, 3]})
dump.to_dict()  # {'endpoint': 'some_api_endpoint', 'data': {'foo': 1, 'bar': '2'}, 'undefined_field_name': [1, 2, 3]}
```

Notes:
- When using `Undefined.INCLUDE`, an `UndefinedParameterError` will be raised if you don't specify
exactly one field of type `CatchAll`.
- Note that `LetterCase` does not affect values written into the `CatchAll` field, they will be as they are given.
- When specifying a default (or a default factory) for the the `CatchAll`-field, e.g. `unknown_things: CatchAll = None`, the default value will be used instead of an empty dict if there are no undefined parameters.
- Calling __init__ with non-keyword arguments resolves the arguments to the defined fields and writes everything else into the catch-all field.

4. All 3 options work as well using `schema().loads` and `schema().dumps`, as long as you don't overwrite it by specifying `schema(unknown=<a marshmallow value>)`.
marshmallow uses the same 3 keywords ['include', 'exclude', 'raise'](https://marshmallow.readthedocs.io/en/stable/quickstart.html#handling-unknown-fields).

5. All 3 operations work as well using `__init__`, e.g. `UnknownAPIDump(**dump_dict)` will **not** raise a `TypeError`, but write all unknown values to the field tagged as `CatchAll`.
   Classes tagged with `EXCLUDE` will also simply ignore unknown parameters. Note that classes tagged as `RAISE` still raise a `TypeError`, and **not** a `UndefinedParameterError` if supplied with unknown keywords.


### Override the default encode / decode / marshmallow field of a specific field?

See [Overriding](#Overriding)

### Handle recursive dataclasses?
Object hierarchies where fields are of the type that they are declared within require a small
type hinting trick to declare the forward reference.
```python
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Tree():
    value: str
    left: Optional['Tree']
    right: Optional['Tree']
```

Avoid using
```python
from __future__ import annotations
```
as it will cause problems with the way dataclasses_json accesses the type annotations.

### Decode classes from None?
Your generic classes should inherit the `OptionalABC` class and
have `__empty__` class method implemented.

An example:
```
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin, OptionalABC

class A:
    def __init__(self, data):
        self.is_empty = data is None
        self.data = data
    def __repr__(self):
        return f'{type(self).__name__}(is_empty={self.is_empty}, data={self.data!r})'

class B(A, OptionalABC):
    @classmethod
    def __empty__(cls):
        return cls(None)

@dataclass
class C(DataClassJsonMixin):
    a: A
    b: B

print(C.from_json('''{"a": 1, "b": 2}'''))
# C(a=1, b=2)

print(C.from_json('''{"a": null, "b": null}'''))
# C(a=None, b=B(is_empty=True, data=None))
```


### Encode/decode my own generic classes?
*This might not work on Python 3.6 properly.*

Your generic classes should inherit the `DecodableGenericABC` class and
have `__encode__` and `__decode__` methods implemented.

A shorter example (for primitive cases):
```
from typing import Generic, TypeVar, List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin, DecodableGenericABC

T = TypeVar('T')
class MyGeneric(Generic[T], DecodableGenericABC):
    _data: List[T]
    def __init__(self, data: List[T]):
        self._data = data
    def __repr__(self):
        return f'MyGeneric(data={self._data!r})'
    def __encode__(self, **kwargs):
        return self._data
    @classmethod
    def __decode__(cls, data, *types, **kwargs):
        return cls(data)

@dataclass
class MyContainer(DataClassJsonMixin):
    int_list: MyGeneric[int]
    str_list: MyGeneric[str]

a = MyGeneric([1, 2, 3])
b = MyGeneric(['a', 'b'])

c = MyContainer(a, b)
print(c.to_json())
# {"int_list": [1, 2, 3], "str_list": ["a", "b"]}

d = MyContainer.from_json('''{"int_list": [1, 2, 3], "str_list": ["a", "b"]}''')
print(d)
# MyContainer(int_list=MyGeneric(data=[1, 2, 3]), str_list=MyGeneric(data=['a', 'b']))
```
An example above does not utilize the `data_encoder` / `data_decoder`
keyword-onnly parameters for the `__encode__` / `__decode__` methods,
and the decoder does not take respect of the `*types` argument.
This works for the primitive classes like `int` or `str`,
but would fail for any classes those require encoding/decoding.

A larger example (for real cases):
```python
from dataclasses import dataclass
from typing import *
from uuid import uuid4

from dataclasses_json import DataClassJsonMixin, DecodableGenericABC, OptionalABC
from dataclasses_json.core import Json

T = TypeVar('T')
K = TypeVar('K')

_MISSING = object()
class CustomMapping(Mapping[K, T], Container[K], Generic[K, T], DecodableGenericABC, num_args=2):
    __slots__ = ('_data', '_id')
    
    _data: Dict[K, T]
    _id: str
    
    def __init__(self, data: Dict[K, T], *, id: str = None):
        self._data = dict(data)
        self._id = id or str(uuid4())
    
    def __getitem__(self, item: K) -> T:
        return self._data[item]
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[K]:
        return iter(self._data.keys())
    
    def __bool__(self) -> bool:
        return bool(self._data)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(id={self._id!r}, data={self._data!r})'
    
    def __eq__(self, other):
        if (not isinstance(other, type(self))):
            raise NotImplementedError
        
        return self._id == other._id and self._data == other._data
    
    def __hash__(self):
        return hash(self._id) + hash(self._data)
    
    @property
    def id(self) -> str:
        return self._id
    
    def __encode__(self, *, data_encoder: Callable[[T], Json], **kwargs) -> Json:
        data_encoded = data_encoder(self._data)
        id_encoded = data_encoder(self._id)
        return dict(id=id_encoded, data=data_encoded)
    
    @classmethod
    def __decode__(cls, data: Json, *types, data_decoder: Callable[[Type[T], Json], T], **kwargs) -> 'CustomMapping[K, T]':
        if (not isinstance(data, Mapping)):
            raise TypeError(f"'data' is expected to be Mapping, got {type(data)}")
        
        k_type, v_type = types
        data = dict(data)
        _data = data_decoder(v_type, data.pop('data'))
        _id = data.pop('id', _MISSING)
        if (_id is not _MISSING):
            _id = data_decoder(str, _id)
        else:
            _id = None
        
        if (data):
            raise ValueError(f"Got unexpected fields while decoding CustomMapping: {list(data.keys())}")
        
        return cls(data=_data, id=_id)

class OptionContainer(Collection[T], Generic[T], DecodableGenericABC, OptionalABC):
    __slots__ = ('_is_empty', '_data')
    
    _is_empty: bool
    _data: T
    
    def __init__(self, data: Optional[T]):
        self._data = data
        self._is_empty = data is None
    
    @classmethod
    def __non_empty__(cls, data: T) -> 'OptionContainer[T]':
        r = cls(data)
        r._is_empty = False
        return r
    
    @classmethod
    def __empty__(cls) -> 'OptionContainer[T]':
        r = cls(None)
        r._is_empty = True
        return r
    
    def __contains__(self, item: T) -> bool:
        return not self._is_empty and item == self._data
    
    def __iter__(self) -> Iterator[T]:
        if (not self._is_empty):
            yield self._data
    
    def __len__(self) -> int:
        return int(self._is_empty)
    
    def __bool__(self) -> bool:
        return self._is_empty
    
    def __repr__(self) -> str:
        if (self._is_empty):
            return f'{self.__class__.__name__}<empty>'
        else:
            return f'{self.__class__.__name__}({self._data!r})'
    
    def __eq__(self, other):
        if (not isinstance(other, type(self))):
            raise NotImplementedError
        
        if (self._is_empty):
            return other._is_empty
        elif (not other._is_empty):
            return self._data == other._data
        else:
            return False
    
    def __hash__(self):
        if (self._is_empty):
            return _EMPTY_HASH__OC
        else:
            return hash(self._data)
    
    def __encode__(self, *, data_encoder: Callable[[T], Json], **kwargs) -> Json:
        if (self._is_empty):
            return None
        else:
            return data_encoder(self._data)
    
    @classmethod
    def __decode__(cls, data: Json, *types: Type[T], data_decoder: Callable[[Type[T], Json], T], **kwargs) -> 'OptionContainer[T]':
        if (data is None):
            return OptionContainer.__empty__()
        else:
            return OptionContainer.__non_empty__(data_decoder(*types, data))
    
    init_empty = __empty__
    init_non_empty = __non_empty__

_EMPTY_HASH__OC = hash(None) + hash(OptionContainer)

@dataclass
class MyClass(DataClassJsonMixin):
    opt_int: OptionContainer[int]
    opt_str: OptionContainer[str]
    opt_mapping: OptionContainer[CustomMapping[str, int]]
```

Test it:
```
opt_a = OptionContainer.init_non_empty(124)
opt_b = OptionContainer.init_empty()
opt_c = OptionContainer(CustomMapping(dict(f1=1, f2=2), id='135121-231566677'))

my_cls = MyClass(opt_a, opt_b, opt_c)
print(my_cls)

actual_json = my_cls.to_json()
print(actual_json)
# {"opt_int": 124, "opt_str": null, "opt_mapping": {"id": "135121-231566677", "data": {"f1": 1, "f2": 2}}}

wanted_json = '''{"opt_int": 124, "opt_str": null, "opt_mapping": {"id": "135121-231566677", "data": {"f1": 1, "f2": 2}}}'''
decoded = my_cls.from_json(wanted_json)
print(decoded)
# MyClass(opt_int=OptionContainer(124), opt_str=OptionContainer<empty>, opt_mapping=OptionContainer(CustomMapping(id='135121-231566677', data={'f1': 1, 'f2': 2})))

assert actual_json == wanted_json, "Encoding failed"
assert decoded == my_cls, "Decoding failed"
```

## Marshmallow interop

Using the `dataclass_json` decorator or mixing in `DataClassJsonMixin` will
provide you with an additional method `.schema()`.

`.schema()` generates a schema exactly equivalent to manually creating a
marshmallow schema for your dataclass. You can reference the [marshmallow API docs](https://marshmallow.readthedocs.io/en/3.0/api_reference.html#schema)
to learn other ways you can use the schema returned by `.schema()`.

You can pass in the exact same arguments to `.schema()` that you would when
constructing a `PersonSchema` instance, e.g. `.schema(many=True)`, and they will
get passed through to the marshmallow schema.


```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Person:
    name: str

# You don't need to do this - it's generated for you by `.schema()`!
from marshmallow import Schema, fields

class PersonSchema(Schema):
    name = fields.Str()
```

Briefly, on what's going on under the hood in the above examples: calling 
`.schema()` will have this library generate a
[marshmallow schema]('https://marshmallow.readthedocs.io/en/3.0/api_reference.html#schema)
for you. It also fills in the corresponding object hook, so that marshmallow
will create an instance of your Data Class on `load` (e.g.
`Person.schema().load` returns a `Person`) rather than a `dict`, which it does
by default in marshmallow.

**Performance note**

`.schema()` is not cached (it generates the schema on every call), so if you
have a nested Data Class you may want to save the result to a variable to 
avoid re-generation of the schema on every usage.

```python
person_schema = Person.schema()
person_schema.dump(people, many=True)

# later in the code...

person_schema.dump(person)
```

## Overriding / Extending

#### Overriding

For example, you might want to encode/decode `datetime` objects using ISO format
rather than the default `timestamp`.

```python
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from datetime import datetime
from marshmallow import fields

@dataclass_json
@dataclass
class DataClassWithIsoDatetime:
    created_at: datetime = field(
        metadata=config(
            encoder=datetime.isoformat,
            decoder=datetime.fromisoformat,
            mm_field=fields.DateTime(format='iso')
        )
    )
```

#### Extending

Similarly, you might want to extend `dataclasses_json` to encode `date` objects.

```python
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from datetime import date
from marshmallow import fields

@dataclass_json
@dataclass
class DataClassWithIsoDatetime:
    created_at: date = field(
        metadata=config(
            encoder= date.isoformat,
            decoder= date.fromisoformat,
            mm_field= fields.DateTime(format='iso')
        ))
```

As you can see, you can **override** or **extend** the default codecs by providing a "hook" via a 
callable:
- `encoder`: a callable, which will be invoked to convert the field value when encoding to JSON
- `decoder`: a callable, which will be invoked to convert the JSON value when decoding from JSON
- `mm_field`: a marshmallow field, which will affect the behavior of any operations involving `.schema()`

Note that these hooks will be invoked regardless if you're using 
`.to_json`/`dump`/`dumps`
and `.from_json`/`load`/`loads`. So apply overrides / extensions judiciously, making sure to 
carefully consider whether the interaction of the encode/decode/mm_field is consistent with what you expect!


#### What if I have other dataclass field extensions that rely on `metadata`

All the `dataclasses_json.config` does is return a mapping, namespaced under the key `'dataclasses_json'`.

Say there's another module, `other_dataclass_package` that uses metadata. Here's how you solve your problem:

```python
metadata = {'other_dataclass_package': 'some metadata...'}  # pre-existing metadata for another dataclass package
dataclass_json_config = config(
            encoder=datetime.isoformat,
            decoder=datetime.fromisoformat,
            mm_field=fields.DateTime(format='iso')
        )
metadata.update(dataclass_json_config)

@dataclass_json
@dataclass
class DataClassWithIsoDatetime:
    created_at: datetime = field(metadata=metadata)
```

You can also manually specify the dataclass_json configuration mapping.

```python
@dataclass_json
@dataclass
class DataClassWithIsoDatetime:
    created_at: date = field(
        metadata={'dataclasses_json': {
            'encoder': date.isoformat,
            'decoder': date.fromisoformat,
            'mm_field': fields.DateTime(format='iso')
        }}
    )
```

## A larger example

```python
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from typing import List

@dataclass_json
@dataclass(frozen=True)
class Minion:
    name: str


@dataclass_json
@dataclass(frozen=True)
class Boss:
    minions: List[Minion]

boss = Boss([Minion('evil minion'), Minion('very evil minion')])
boss_json = """
{
    "minions": [
        {
            "name": "evil minion"
        },
        {
            "name": "very evil minion"
        }
    ]
}
""".strip()

assert boss.to_json(indent=4) == boss_json
assert Boss.from_json(boss_json) == boss
```

## Performance

Take a look at [this issue](https://github.com/lidatong/dataclasses-json/issues/228)

## Versioning

Note this library is still pre-1.0.0 (SEMVER).

The current convention is:
- **PATCH** version upgrades for bug fixes and minor feature additions.
- **MINOR** version upgrades for big API features and breaking changes.

Once this library is 1.0.0, it will follow standard SEMVER conventions.


## Roadmap

Currently the focus is on investigating and fixing bugs in this library, working
on performance, and finishing [this issue](https://github.com/lidatong/dataclasses-json/issues/31).

That said, if you think there's a feature missing / something new needed in the
library, please see the contributing section below.


## Contributing

First of all, thank you for being interested in contributing to this library.
I really appreciate you taking the time to work on this project.

- If you're just interested in getting into the code, a good place to start are 
issues tagged as bugs.
- If introducing a new feature, especially one that modifies the public API, 
consider submitting an issue for discussion before a PR. Please also take a look 
at existing issues / PRs to see what you're proposing has  already been covered 
before / exists.
- I like to follow the commit conventions documented [here](https://www.conventionalcommits.org/en/v1.0.0/#summary)
