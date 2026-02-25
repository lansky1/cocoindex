"""Milvus vector database target implementation."""

import asyncio
import dataclasses
import json
import logging
import threading
import weakref
from typing import Any, Literal

from pymilvus import (  # type: ignore
    MilvusClient,
    Collection,
    CollectionSchema,
    FieldSchema as MilvusFieldSchema,
)

from cocoindex import op
from cocoindex.auth_registry import AuthEntryReference, get_auth_entry
from cocoindex.engine_type import FieldSchema, BasicValueType
from cocoindex.index import (
    IndexOptions,
    VectorSimilarityMetric,
    VectorIndexMethod,
    HnswVectorIndexMethod,
    IvfFlatVectorIndexMethod,
)

_logger = logging.getLogger(__name__)

_MILVUS_METRIC_MAP: dict[VectorSimilarityMetric, str] = {
    VectorSimilarityMetric.COSINE_SIMILARITY: "cosine",
    VectorSimilarityMetric.L2_DISTANCE: "l2",
    VectorSimilarityMetric.INNER_PRODUCT: "ip",
}


@dataclasses.dataclass
class MilvusConnection:
    """Connection spec for Milvus."""

    host: str
    """Milvus server host (e.g., "localhost" or IP address)"""
    port: int
    """Milvus server port (default 19530 for gRPC, 9091 for HTTP)"""
    api_key: str | None = None
    """Optional API key for authentication"""
    db_name: str = "default"
    """Database name (default: "default")"""
    use_http: bool = False
    """Use HTTP connection instead of gRPC (default: False)"""


class Milvus(op.TargetSpec):
    """Target powered by Milvus - https://milvus.io/."""

    collection_name: str
    """Name of the Milvus collection to store data"""
    connection: AuthEntryReference[MilvusConnection]
    """Connection reference to Milvus server"""
    consistency_level: Literal["STRONG", "BOUNDED", "EVENTUALLY", "CUSTOMIZED"] = "EVENTUALLY"
    """Consistency level for reads and writes"""


@dataclasses.dataclass
class _PersistentKey:
    """Persistent key for Milvus collection."""

    collection_name: str
    db_name: str
    uri: str
    connection_ref: AuthEntryReference[MilvusConnection]
    """Reference to connection credentials"""


@dataclasses.dataclass
class _VectorIndexInfo:
    """Information about a vector index."""

    field_name: str
    metric: str
    method: VectorIndexMethod | None = None


@dataclasses.dataclass
class _State:
    """Setup state for Milvus target."""

    collection_name: str
    db_name: str
    key_field_schema: FieldSchema
    value_fields_schema: list[FieldSchema]
    vector_index_info: _VectorIndexInfo | None = None


@dataclasses.dataclass
class _MutateContext:
    """Context for applying mutations to Milvus."""

    client: MilvusClient
    collection: Collection
    collection_name: str
    key_field_name: str
    key_field_schema: FieldSchema
    value_fields_schema: list[FieldSchema]


def _is_vector_field(field: FieldSchema) -> bool:
    """Check if a field is a vector field."""
    value_type = field.value_type.type
    if isinstance(value_type, BasicValueType):
        return value_type.kind == "Vector"
    return False


def _get_vector_field_info(
    value_fields_schema: list[FieldSchema],
) -> tuple[FieldSchema, int]:
    """
    Extract vector field from value schema.

    Returns:
        Tuple of (field_schema, dimension)

    Raises:
        ValueError: If no vector field found or dimension missing
    """
    vector_fields = [f for f in value_fields_schema if _is_vector_field(f)]

    if not vector_fields:
        raise ValueError(
            "Milvus requires at least one vector field in the value schema"
        )
    if len(vector_fields) > 1:
        raise ValueError(
            f"Milvus target supports single vector field only, "
            f"but found {len(vector_fields)}: {[f.name for f in vector_fields]}. "
            f"Consider using LanceDB for multiple vector fields."
        )

    vector_field = vector_fields[0]
    dimension = _get_vector_dimension(vector_field)

    return vector_field, dimension


def _get_vector_dimension(field: FieldSchema) -> int:
    """
    Extract vector dimension from field schema.

    Raises:
        ValueError: If dimension is not specified
    """
    value_type = field.value_type.type
    if isinstance(value_type, BasicValueType) and value_type.kind == "Vector":
        if value_type.vector is None:
            raise ValueError(f"Vector field {field.name} missing vector schema")
        if value_type.vector.dimension is None:
            raise ValueError(
                f"Vector field {field.name} missing dimension specification"
            )
        return value_type.vector.dimension

    raise ValueError(f"Field {field.name} is not a vector field")


def _cocoindex_type_to_milvus_field_type(field: FieldSchema, is_key: bool = False) -> str:
    """Convert CocoIndex type to Milvus field type."""
    value_type = field.value_type.type

    if isinstance(value_type, BasicValueType):
        kind = value_type.kind

        if kind == "Vector":
            if value_type.vector is None:
                raise ValueError(f"Vector field {field.name} missing vector schema")
            if value_type.vector.dimension is None:
                raise ValueError(
                    f"Vector field {field.name} missing dimension specification"
                )
            return "FloatVector"

        type_mapping = {
            "Bytes": "BinaryVector",
            "Str": "VarChar",
            "Bool": "Bool",
            "Int64": "Int64",
            "Float32": "Float",
            "Float64": "Double",
            "Uuid": "VarChar",
            "Date": "VarChar",
            "Time": "VarChar",
            "LocalDateTime": "VarChar",
            "OffsetDateTime": "VarChar",
            "TimeDelta": "Int64",
            "Json": "JSON",
        }

        if kind in type_mapping:
            return type_mapping[kind]

    return "JSON"


def _convert_value_to_milvus(value: Any) -> Any:
    """Convert Python value to Milvus-compatible format."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return value
    return json.dumps(value, default=str)


_ClientsLock = threading.Lock()
_Clients: weakref.WeakValueDictionary[str, MilvusClient] = (
    weakref.WeakValueDictionary()
)


async def get_milvus_client(
    host: str,
    port: int,
    use_http: bool = False,
    api_key: str | None = None,
    db_name: str = "default",
) -> MilvusClient:
    """
    Get or create a Milvus client.

    Uses connection pooling to reuse connections.
    """
    protocol = "http" if use_http else "grpc"
    uri = f"{protocol}://{host}:{port}"
    cache_key = f"{uri}:{db_name}"

    with _ClientsLock:
        client = _Clients.get(cache_key)
        if client is None:
            client = MilvusClient(
                uri=uri,
                token=api_key,
                db_name=db_name,
            )
            _Clients[cache_key] = client

    return client


async def create_collection_schema(
    key_field_schema: FieldSchema,
    value_fields_schema: list[FieldSchema],
    vector_field_name: str,
    vector_dimension: int,
    metric: str,
) -> CollectionSchema:
    """Create a Milvus collection schema."""
    fields = []

    key_field_type = _cocoindex_type_to_milvus_field_type(key_field_schema, is_key=True)
    fields.append(
        MilvusFieldSchema(
            name=key_field_schema.name,
            dtype=key_field_type,
            is_primary=True,
            auto_id=False,
            max_length=256 if key_field_type == "VarChar" else None,
        )
    )

    for field in value_fields_schema:
        if field.name == vector_field_name:
            fields.append(
                MilvusFieldSchema(
                    name=field.name,
                    dtype="FloatVector",
                    dim=vector_dimension,
                )
            )
        else:
            field_type = _cocoindex_type_to_milvus_field_type(field)
            kwargs: dict[str, Any] = {
                "name": field.name,
                "dtype": field_type,
            }
            if field_type == "VarChar":
                kwargs["max_length"] = 65535
            fields.append(MilvusFieldSchema(**kwargs))

    return CollectionSchema(fields=fields)


@op.target_connector(
    spec_cls=Milvus,
    persistent_key_type=_PersistentKey,
    setup_state_cls=_State,
)
class _Connector:
    """Milvus target connector implementation."""

    @staticmethod
    def get_persistent_key(spec: Milvus) -> _PersistentKey:
        """Get persistent key for the Milvus collection."""
        conn = get_auth_entry(MilvusConnection, spec.connection)
        return _PersistentKey(
            collection_name=spec.collection_name,
            db_name=conn.db_name,
            uri=f"{conn.host}:{conn.port}",
            connection_ref=spec.connection,
        )

    @staticmethod
    def get_setup_state(
        spec: Milvus,
        key_fields_schema: list[FieldSchema],
        value_fields_schema: list[FieldSchema],
        index_options: IndexOptions,
    ) -> _State:
        """Build setup state from spec and schema information."""
        if len(key_fields_schema) != 1:
            raise ValueError("Milvus only supports a single key field")
        
        conn = get_auth_entry(MilvusConnection, spec.connection)

        vector_field, vector_dimension = _get_vector_field_info(value_fields_schema)

        if not index_options.vector_indexes:
            raise ValueError(
                "Milvus requires at least one vector index in index_options"
            )
        if len(index_options.vector_indexes) > 1:
            raise ValueError(
                "Milvus supports single vector index only, "
                f"but found {len(index_options.vector_indexes)}"
            )

        vector_index = index_options.vector_indexes[0]

        if vector_index.field_name != vector_field.name:
            raise ValueError(
                f"Vector index field '{vector_index.field_name}' "
                f"does not match vector field '{vector_field.name}'"
            )

        metric = _MILVUS_METRIC_MAP.get(vector_index.metric, "cosine")

        vector_index_info = _VectorIndexInfo(
            field_name=vector_field.name,
            metric=metric,
            method=vector_index.method,
        )

        return _State(
            collection_name=spec.collection_name,
            db_name=conn.db_name,
            key_field_schema=key_fields_schema[0],
            value_fields_schema=value_fields_schema,
            vector_index_info=vector_index_info,
        )

    @staticmethod
    def describe(key: _PersistentKey) -> str:
        """Get human-readable description of the target."""
        return (
            f"Milvus collection '{key.collection_name}' "
            f"in database '{key.db_name}' at {key.uri}"
        )

    @staticmethod
    def check_state_compatibility(
        previous: _State, current: _State
    ) -> op.TargetStateCompatibility:
        """Check if schema changes are compatible."""
        if previous.key_field_schema != current.key_field_schema:
            return op.TargetStateCompatibility.NOT_COMPATIBLE

        if len(previous.value_fields_schema) != len(current.value_fields_schema):
            return op.TargetStateCompatibility.NOT_COMPATIBLE

        for prev_field, curr_field in zip(
            previous.value_fields_schema, current.value_fields_schema
        ):
            if prev_field.name != curr_field.name:
                return op.TargetStateCompatibility.NOT_COMPATIBLE
            if prev_field.value_type != curr_field.value_type:
                return op.TargetStateCompatibility.NOT_COMPATIBLE

        if previous.vector_index_info and current.vector_index_info:
            prev_dim = _get_vector_dimension(
                next(
                    f
                    for f in previous.value_fields_schema
                    if f.name == previous.vector_index_info.field_name
                )
            )
            curr_dim = _get_vector_dimension(
                next(
                    f
                    for f in current.value_fields_schema
                    if f.name == current.vector_index_info.field_name
                )
            )
            if prev_dim != curr_dim:
                return op.TargetStateCompatibility.NOT_COMPATIBLE

        return op.TargetStateCompatibility.COMPATIBLE

    @staticmethod
    async def apply_setup_change(
        key: _PersistentKey, previous: _State | None, current: _State | None
    ) -> None:
        """Apply setup changes (create/drop collection and indexes)."""
        if previous is None and current is None:
            return

        latest_state = current or previous
        if latest_state is None:
            return

        conn = get_auth_entry(MilvusConnection, key.connection_ref)
        client = await get_milvus_client(
            host=conn.host,
            port=conn.port,
            use_http=conn.use_http,
            api_key=conn.api_key,
            db_name=conn.db_name,
        )

        try:
            collections = client.list_collections()

            if previous is None and current is not None:
                if current.collection_name not in collections:
                    vector_field = next(
                        f
                        for f in current.value_fields_schema
                        if _is_vector_field(f)
                    )
                    vector_dim = _get_vector_dimension(vector_field)

                    schema = await create_collection_schema(
                        current.key_field_schema,
                        current.value_fields_schema,
                        vector_field.name,
                        vector_dim,
                        current.vector_index_info.metric if current.vector_index_info else "cosine",
                    )

                    client.create_collection(
                        collection_name=current.collection_name,
                        schema=schema,
                    )
                    _logger.info(
                        "Created Milvus collection '%s' in database '%s'",
                        current.collection_name,
                        current.db_name,
                    )

                    if current.vector_index_info:
                        params: dict[str, Any] = {"M": 16, "efConstruction": 200}
                        index_params: dict[str, Any] = {
                            "index_type": "HNSW",
                            "metric_type": current.vector_index_info.metric.upper(),
                            "params": params,
                        }

                        if isinstance(current.vector_index_info.method, HnswVectorIndexMethod):
                            if current.vector_index_info.method.m is not None:
                                params["M"] = current.vector_index_info.method.m
                            if current.vector_index_info.method.ef_construction is not None:
                                params["efConstruction"] = current.vector_index_info.method.ef_construction
                        elif isinstance(current.vector_index_info.method, IvfFlatVectorIndexMethod):
                            index_params["index_type"] = "IVF_FLAT"
                            if current.vector_index_info.method.lists is not None:
                                params = {"nlist": current.vector_index_info.method.lists}
                                index_params["params"] = params

                        client.create_index(
                            collection_name=current.collection_name,
                            field_name=current.vector_index_info.field_name,
                            index_params=index_params,
                        )
                        _logger.info(
                            "Created index on field '%s' for collection '%s'",
                            current.vector_index_info.field_name,
                            current.collection_name,
                        )

            elif previous is not None and current is None:
                if previous.collection_name in collections:
                    client.drop_collection(collection_name=previous.collection_name)
                    _logger.info(
                        "Dropped Milvus collection '%s' from database '%s'",
                        previous.collection_name,
                        previous.db_name,
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            _logger.error(
                "Error applying setup changes to Milvus collection '%s': %s",
                key.collection_name,
                e,
                exc_info=True,
            )
            raise

    @staticmethod
    async def prepare(
        spec: Milvus,
        setup_state: _State,
    ) -> _MutateContext:
        """Prepare for mutations - get collection reference and context."""
        conn = get_auth_entry(MilvusConnection, spec.connection)
        
        client = await get_milvus_client(
            host=conn.host,
            port=conn.port,
            use_http=conn.use_http,
            api_key=conn.api_key,
            db_name=conn.db_name,
        )

        try:
            collections = client.list_collections()
            if spec.collection_name not in collections:
                vector_field = next(
                    f
                    for f in setup_state.value_fields_schema
                    if _is_vector_field(f)
                )
                vector_dim = _get_vector_dimension(vector_field)

                schema = await create_collection_schema(
                    setup_state.key_field_schema,
                    setup_state.value_fields_schema,
                    vector_field.name,
                    vector_dim,
                    setup_state.vector_index_info.metric if setup_state.vector_index_info else "cosine",
                )

                client.create_collection(
                    collection_name=spec.collection_name,
                    schema=schema,
                )

                if setup_state.vector_index_info:
                    params: dict[str, Any] = {"M": 16, "efConstruction": 200}
                    index_params: dict[str, Any] = {
                        "index_type": "HNSW", 
                        "metric_type": setup_state.vector_index_info.metric.upper(),
                        "params": params,
                    }

                    if isinstance(setup_state.vector_index_info.method, HnswVectorIndexMethod):
                        if setup_state.vector_index_info.method.m is not None:
                            params["M"] = setup_state.vector_index_info.method.m
                        if setup_state.vector_index_info.method.ef_construction is not None:
                            params["efConstruction"] = setup_state.vector_index_info.method.ef_construction
                    elif isinstance(setup_state.vector_index_info.method, IvfFlatVectorIndexMethod):
                        index_params["index_type"] = "IVF_FLAT"
                        if setup_state.vector_index_info.method.lists is not None:
                            params = {"nlist": setup_state.vector_index_info.method.lists}
                            index_params["params"] = params

                    client.create_index(
                        collection_name=spec.collection_name,
                        field_name=setup_state.vector_index_info.field_name,
                        index_params=index_params,
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            _logger.error("Error preparing Milvus collection: %s", e, exc_info=True)
            raise

        try:
            collection = Collection(
                name=spec.collection_name,
                using=client._using,  # type: ignore
            )
        except Exception as e:
            _logger.error("Error getting collection reference: %s", e)
            raise

        return _MutateContext(
            client=client,
            collection=collection,
            collection_name=spec.collection_name,
            key_field_name=setup_state.key_field_schema.name,
            key_field_schema=setup_state.key_field_schema,
            value_fields_schema=setup_state.value_fields_schema,
        )

    @staticmethod
    async def mutate(
        *all_mutations: tuple[_MutateContext, dict[Any, dict[str, Any] | None]],
    ) -> None:
        """Apply mutations (upserts and deletes) to Milvus.
        
        Note: Milvus does not support transactional semantics like traditional SQL databases.
        While we execute both deletes and upserts within a single executor call, they are
        not guaranteed to be atomic. We perform deletes first to minimize the risk of
        conflicts if an operation fails mid-transaction.
        """
        for context, mutations in all_mutations:
            if not mutations:
                continue

            upsert_rows = []
            delete_ids = []

            key_field_name = context.key_field_name

            for key, value in mutations.items():
                if value is None:
                    delete_ids.append(key)
                else:
                    row = {key_field_name: key}
                    for field_name, field_value in value.items():
                        row[field_name] = _convert_value_to_milvus(field_value)
                    upsert_rows.append(row)

            def _do_mutations() -> None:
                if delete_ids:
                    try:
                        ids_str = ", ".join(str(id) for id in delete_ids)
                        filter_expr = f"{key_field_name} in [{ids_str}]"
                        context.client.delete(
                            collection_name=context.collection_name,
                            filter=filter_expr,
                        )
                    except Exception as e:
                        _logger.error(
                            "Error deleting records from Milvus: %s", e, exc_info=True
                        )
                        raise

                if upsert_rows:
                    try:
                        context.client.upsert(
                            collection_name=context.collection_name,
                            records=upsert_rows,
                        )
                    except Exception as e:
                        _logger.error(
                            "Error upserting records to Milvus: %s", e, exc_info=True
                        )
                        raise

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_mutations)

    @staticmethod
    async def cleanup(context: _MutateContext) -> None:
        """Cleanup connection."""
        pass

