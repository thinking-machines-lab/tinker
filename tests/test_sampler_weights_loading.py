"""Tests for sampler_weights checkpoint loading (Issue #25)."""

import pytest

from tinker.types.checkpoint import ParsedCheckpointTinkerPath


class TestParsedCheckpointTinkerPath:
    """Tests for ParsedCheckpointTinkerPath to ensure sampler_weights loading works correctly."""

    def test_parse_weights_checkpoint(self) -> None:
        """Test parsing a weights checkpoint path."""
        parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-123/weights/0001")
        assert parsed.training_run_id == "run-123"
        assert parsed.checkpoint_type == "training"
        assert parsed.checkpoint_id == "0001"
        assert parsed.api_checkpoint_id == "0001"

    def test_parse_sampler_weights_checkpoint(self) -> None:
        """Test parsing a sampler_weights checkpoint path (Issue #25)."""
        parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-123/sampler_weights/0001")
        assert parsed.training_run_id == "run-123"
        assert parsed.checkpoint_type == "sampler"
        assert parsed.checkpoint_id == "0001"
        # This is the key fix: api_checkpoint_id should include sampler_weights prefix
        assert parsed.api_checkpoint_id == "sampler_weights/0001"

    def test_api_checkpoint_id_for_weights(self) -> None:
        """Test that weights checkpoints don't add prefix to checkpoint_id."""
        parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://model-789/weights/final")
        assert parsed.api_checkpoint_id == "final"

    def test_api_checkpoint_id_for_sampler_weights(self) -> None:
        """Test that sampler_weights checkpoints include sampler_weights prefix in api_checkpoint_id."""
        parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://model-789/sampler_weights/ckpt-001")
        assert parsed.api_checkpoint_id == "sampler_weights/ckpt-001"

    def test_invalid_tinker_path_no_prefix(self) -> None:
        """Test that paths without tinker:// prefix are rejected."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("run-123/weights/0001")

    def test_invalid_tinker_path_wrong_type(self) -> None:
        """Test that invalid checkpoint types are rejected."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-123/invalid/0001")

    def test_invalid_tinker_path_wrong_parts(self) -> None:
        """Test that paths with wrong number of parts are rejected."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-123/weights")

    def test_sampler_weights_with_custom_name(self) -> None:
        """Test sampler_weights with custom checkpoint name."""
        parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-456/sampler_weights/my-sampler-v2")
        assert parsed.training_run_id == "run-456"
        assert parsed.checkpoint_type == "sampler"
        assert parsed.checkpoint_id == "my-sampler-v2"
        assert parsed.api_checkpoint_id == "sampler_weights/my-sampler-v2"