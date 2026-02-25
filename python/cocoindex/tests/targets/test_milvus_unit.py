"""
Unit tests for Milvus connector (no database connection required).
"""
# mypy: disable-error-code="no-untyped-def"

from typing import Literal
import pytest

from cocoindex.targets.milvus import (
    Milvus,
    MilvusConnection,
    _State,
    _PersistentKey,
    _VectorIndexInfo,
    _Connector,
    _is_vector_field,
    _get_vector_field_info,
    _get_vector_dimension,
    _cocoindex_type_to_milvus_field_type,
    _convert_value_to_milvus,
    _MILVUS_METRIC_MAP,
)
from cocoindex.auth_registry import AuthEntryReference
from cocoindex.engine_type import (
    FieldSchema,
    EnrichedValueType,
    BasicValueType,
    VectorTypeSchema,
)
from cocoindex import op
from cocoindex.index import (
    IndexOptions,
    VectorIndexDef,
    VectorSimilarityMetric,
)

_BasicKind = Literal[
    "Bytes",
    "Str",
    "Bool",
    "Int64",
    "Float32",
    "Float64",
    "Range",
    "Uuid",
    "Date",
    "Time",
    "LocalDateTime",
    "OffsetDateTime",
    "TimeDelta",
    "Json",
    "Vector",
    "Union",
]


def _mock_field(
    name: str, kind: str | _BasicKind, nullable: bool = False, dim: int | None = None
) -> FieldSchema:
    """Create mock FieldSchema for testing."""
    if kind == "Vector":
        vec_schema = VectorTypeSchema(
            element_type=BasicValueType(kind="Float32"),  # type: ignore
            dimension=dim,
        )
        basic_type = BasicValueType(kind=kind, vector=vec_schema)  # type: ignore
    else:
        basic_type = BasicValueType(kind=kind)  # type: ignore
    return FieldSchema(
        name=name,
        value_type=EnrichedValueType(type=basic_type, nullable=nullable),
    )


class TestHelperFunctions:
    """Test helper functions for Milvus connector."""

    def test_is_vector_field_valid(self):
        """Test _is_vector_field with a vector field."""
        field = _mock_field("embedding", "Vector", dim=384)
        assert _is_vector_field(field) is True

    def test_is_vector_field_invalid(self):
        """Test _is_vector_field with a non-vector field."""
        field = _mock_field("text", "Str")
        assert _is_vector_field(field) is False

    def test_get_vector_field_info_single_vector(self):
        """Test _get_vector_field_info with single vector field."""
        fields = [
            _mock_field("id", "Int64"),
            _mock_field("embedding", "Vector", dim=384),
            _mock_field("text", "Str"),
        ]
        vector_field, dim = _get_vector_field_info(fields)
        assert vector_field.name == "embedding"
        assert dim == 384

    def test_get_vector_field_info_no_vector(self):
        """Test _get_vector_field_info with no vector field raises ValueError."""
        fields = [
            _mock_field("id", "Int64"),
            _mock_field("text", "Str"),
        ]
        with pytest.raises(ValueError, match="requires at least one vector field"):
            _get_vector_field_info(fields)

    def test_get_vector_field_info_multiple_vectors(self):
        """Test _get_vector_field_info with multiple vector fields raises ValueError."""
        fields = [
            _mock_field("embedding1", "Vector", dim=384),
            _mock_field("embedding2", "Vector", dim=768),
        ]
        with pytest.raises(ValueError, match="supports single vector field only"):
            _get_vector_field_info(fields)

    def test_get_vector_dimension_valid(self):
        """Test _get_vector_dimension with valid vector field."""
        field = _mock_field("embedding", "Vector", dim=384)
        dim = _get_vector_dimension(field)
        assert dim == 384

    def test_get_vector_dimension_missing(self):
        """Test _get_vector_dimension with missing dimension raises ValueError."""
        vec_schema = VectorTypeSchema(
            element_type=BasicValueType(kind="Float32"),
            dimension=None,
        )
        basic_type = BasicValueType(kind="Vector", vector=vec_schema)
        field = FieldSchema(
            name="embedding",
            value_type=EnrichedValueType(type=basic_type),
        )
        with pytest.raises(ValueError, match="missing dimension specification"):
            _get_vector_dimension(field)

    def test_cocoindex_type_to_milvus_field_type(self):
        """Test type conversion from CocoIndex to Milvus types."""
        test_cases = [
            ("Str", "VarChar"),
            ("Int64", "Int64"),
            ("Float32", "Float"),
            ("Float64", "Double"),
            ("Bool", "Bool"),
            ("Json", "JSON"),
        ]
        for cocoindex_kind, expected_milvus_type in test_cases:
            field = _mock_field("test_field", cocoindex_kind)
            result = _cocoindex_type_to_milvus_field_type(field)
            assert result == expected_milvus_type, f"Failed for {cocoindex_kind}"

    def test_cocoindex_vector_to_milvus_type(self):
        """Test vector field type conversion."""
        field = _mock_field("embedding", "Vector", dim=384)
        result = _cocoindex_type_to_milvus_field_type(field)
        assert result == "FloatVector"

    def test_convert_value_to_milvus(self):
        """Test value conversion for Milvus storage."""
        test_cases = [
            (None, None),
            ("text", "text"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ({"key": "value"}, {"key": "value"}),
        ]
        for value, expected in test_cases:
            result = _convert_value_to_milvus(value)
            if isinstance(result, str) and not isinstance(value, str):
                import json
                assert json.loads(result) == expected
            else:
                assert result == expected

    def test_metric_mapping(self):
        """Test metric type mapping from CocoIndex to Milvus."""
        assert _MILVUS_METRIC_MAP[VectorSimilarityMetric.COSINE_SIMILARITY] == "cosine"
        assert _MILVUS_METRIC_MAP[VectorSimilarityMetric.L2_DISTANCE] == "l2"
        assert _MILVUS_METRIC_MAP[VectorSimilarityMetric.INNER_PRODUCT] == "ip"


class TestStateValidation:
    """Test state validation and compatibility checking."""

    def _make_state(
        self,
        key_field_name: str = "id",
        vector_field_name: str = "embedding",
        vector_dim: int = 384,
    ) -> _State:
        """Helper to create _State for testing."""
        return _State(
            collection_name="test_collection",
            db_name="default",
            key_field_schema=_mock_field(key_field_name, "Int64"),
            value_fields_schema=[
                _mock_field(vector_field_name, "Vector", dim=vector_dim),
                _mock_field("text", "Str"),
            ],
            vector_index_info=_VectorIndexInfo(
                field_name=vector_field_name,
                metric="cosine",
                method=None,
            ),
        )

    def test_state_compatibility_identical(self):
        """Test that identical states are COMPATIBLE."""
        state = self._make_state()
        assert (
            _Connector.check_state_compatibility(state, state)
            == op.TargetStateCompatibility.COMPATIBLE
        )

    def test_state_compatibility_key_field_changed(self):
        """Test that changing key field makes state NOT_COMPATIBLE."""
        state1 = self._make_state(key_field_name="id")
        state2 = self._make_state(key_field_name="doc_id")
        assert (
            _Connector.check_state_compatibility(state1, state2)
            == op.TargetStateCompatibility.NOT_COMPATIBLE
        )

    def test_state_compatibility_vector_dim_changed(self):
        """Test that changing vector dimension makes state NOT_COMPATIBLE."""
        state1 = self._make_state(vector_dim=384)
        state2 = self._make_state(vector_dim=768)
        assert (
            _Connector.check_state_compatibility(state1, state2)
            == op.TargetStateCompatibility.NOT_COMPATIBLE
        )

    def test_state_compatibility_field_count_changed(self):
        """Test that changing number of fields makes state NOT_COMPATIBLE."""
        state1 = self._make_state()
        state2 = _State(
            collection_name="test_collection",
            db_name="default",
            key_field_schema=_mock_field("id", "Int64"),
            value_fields_schema=[
                _mock_field("embedding", "Vector", dim=384),
            ],
            vector_index_info=_VectorIndexInfo(
                field_name="embedding",
                metric="cosine",
                method=None,
            ),
        )
        assert (
            _Connector.check_state_compatibility(state1, state2)
            == op.TargetStateCompatibility.NOT_COMPATIBLE
        )


class TestSpecValidation:
    """Test Milvus target spec validation."""

    def test_get_setup_state_single_key_field(self):
        """Test get_setup_state requires single key field."""
        key_fields = [
            _mock_field("id", "Int64"),
            _mock_field("type", "Str"),
        ]
        value_fields = [
            _mock_field("embedding", "Vector", dim=384),
        ]
        index_options = IndexOptions(
            primary_key_fields=["id", "type"],
            vector_indexes=[
                VectorIndexDef(
                    field_name="embedding",
                    metric=VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ]
        )

        from unittest.mock import Mock

        spec = Mock(spec=Milvus)
        spec.collection_name = "test"
        spec.connection = Mock()

        with pytest.raises(ValueError, match="single key field"):
            _Connector.get_setup_state(spec, key_fields, value_fields, index_options)

    def test_get_setup_state_requires_vector_field(self):
        """Test get_setup_state requires vector field."""
        from unittest.mock import Mock, patch

        key_fields = [_mock_field("id", "Int64")]
        value_fields = [_mock_field("text", "Str")]
        index_options = IndexOptions(
            primary_key_fields=["id"],
            vector_indexes=[]
        )

        spec = Mock(spec=Milvus)
        spec.collection_name = "test"
        spec.connection = Mock()

        with patch("cocoindex.targets.milvus.get_auth_entry") as mock_get_auth:
            mock_get_auth.return_value = MilvusConnection(
                host="localhost",
                port=19530,
                db_name="default",
            )
            with pytest.raises(ValueError, match="requires at least one vector field"):
                _Connector.get_setup_state(spec, key_fields, value_fields, index_options)

    def test_get_setup_state_vector_index_mismatch(self):
        """Test get_setup_state validates vector index matches vector field."""
        from unittest.mock import Mock, patch

        key_fields = [_mock_field("id", "Int64")]
        value_fields = [_mock_field("embedding", "Vector", dim=384)]
        index_options = IndexOptions(
            primary_key_fields=["id"],
            vector_indexes=[
                VectorIndexDef(
                    field_name="wrong_field",
                    metric=VectorSimilarityMetric.COSINE_SIMILARITY,
                )
            ]
        )

        spec = Mock(spec=Milvus)
        spec.collection_name = "test"
        spec.connection = Mock()

        with patch("cocoindex.targets.milvus.get_auth_entry") as mock_get_auth:
            mock_get_auth.return_value = MilvusConnection(
                host="localhost",
                port=19530,
                db_name="default",
            )
            with pytest.raises(ValueError, match="does not match vector field"):
                _Connector.get_setup_state(spec, key_fields, value_fields, index_options)


class TestPersistentKeyGeneration:
    """Test persistent key generation."""

    def test_get_persistent_key_format(self):
        """Test _PersistentKey has correct format."""
        from unittest.mock import Mock
        
        mock_conn_ref = Mock(spec=AuthEntryReference)
        key = _PersistentKey(
            collection_name="my_collection",
            db_name="default",
            uri="localhost:19530",
            connection_ref=mock_conn_ref,
        )
        assert key.collection_name == "my_collection"
        assert key.db_name == "default"
        assert key.uri == "localhost:19530"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
