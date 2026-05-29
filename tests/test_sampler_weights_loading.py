"""Tests for sampler_weights checkpoint loading functionality."""

import pytest
from tinker.types.checkpoint import ParsedCheckpointTinkerPath


class TestParsedCheckpointTinkerPath:
    """Test parsing of checkpoint tinker paths."""

    def test_parse_weights_checkpoint_path(self):
        """Test parsing a weights checkpoint path."""
        path = "tinker://run-id123/weights/0001"
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(path)
        
        assert parsed.tinker_path == path
        assert parsed.training_run_id == "run-id123"
        assert parsed.checkpoint_type == "training"
        assert parsed.checkpoint_id == "weights/0001"
        assert parsed.api_checkpoint_id == "0001"

    def test_parse_sampler_weights_checkpoint_path(self):
        """Test parsing a sampler_weights checkpoint path."""
        path = "tinker://run-id456/sampler_weights/sampler-001"
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(path)
        
        assert parsed.tinker_path == path
        assert parsed.training_run_id == "run-id456"
        assert parsed.checkpoint_type == "sampler"
        assert parsed.checkpoint_id == "sampler_weights/sampler-001"
        assert parsed.api_checkpoint_id == "sampler_weights/sampler-001"

    def test_api_checkpoint_id_for_training(self):
        """Test api_checkpoint_id returns correct format for training checkpoints."""
        path = "tinker://abc123/weights/0100"
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(path)
        
        # For training, should return just the checkpoint number
        assert parsed.api_checkpoint_id == "0100"

    def test_api_checkpoint_id_for_sampler(self):
        """Test api_checkpoint_id returns correct format for sampler checkpoints."""
        path = "tinker://xyz789/sampler_weights/final-model"
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(path)
        
        # For sampler, should return the full prefixed ID
        assert parsed.api_checkpoint_id == "sampler_weights/final-model"

    def test_invalid_tinker_path_no_prefix(self):
        """Test that invalid paths without tinker:// prefix raise error."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("run-id/weights/0001")

    def test_invalid_tinker_path_wrong_parts(self):
        """Test that paths with wrong number of parts raise error."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-id/weights")

    def test_invalid_checkpoint_type(self):
        """Test that invalid checkpoint type raises error."""
        with pytest.raises(ValueError, match="Invalid tinker path"):
            ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-id/invalid/0001")