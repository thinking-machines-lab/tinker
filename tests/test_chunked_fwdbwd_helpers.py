import pytest

from tinker.lib.chunked_fwdbwd_helpers import (
    _unique,
    _metrics_reduction,
    REDUCE_MAP,
)
from tinker.types import ForwardBackwardOutput, LossFnOutput, TensorData


class TestMetricsReductionWithUnique:
    """Test the _metrics_reduction function with unique metrics."""

    def create_forward_backward_output(self, metrics: dict, num_loss_fn_outputs: int = 1) -> ForwardBackwardOutput:
        """Helper to create ForwardBackwardOutput for testing."""
        # LossFnOutput is Dict[str, TensorData], so create sample data
        tensor_data = TensorData(
            data=[0.0],
            dtype="float32",
            shape=[1]
        )
        loss_fn_outputs = [{"loss": tensor_data} for _ in range(num_loss_fn_outputs)]
        return ForwardBackwardOutput(
            loss_fn_output_type="test",
            metrics=metrics,
            loss_fn_outputs=loss_fn_outputs,
        )

    def test_unique_reduction_single_result(self):
        """Test unique reduction with single result."""
        results = [
            self.create_forward_backward_output({"clock_cycle:unique": 12345})
        ]

        reduced = _metrics_reduction(results)

        assert "clock_cycle:unique" in reduced
        assert reduced["clock_cycle:unique"] == 12345

    def test_unique_reduction_multiple_results(self):
        """Test unique reduction with multiple results creates additional keys."""
        results = [
            self.create_forward_backward_output({"clock_cycle:unique": 12345}),
            self.create_forward_backward_output({"clock_cycle:unique": 12346}),
            self.create_forward_backward_output({"clock_cycle:unique": 12347}),
        ]

        reduced = _metrics_reduction(results)

        # First value should be the main key
        assert reduced["clock_cycle:unique"] == 12345
        # Additional values should get suffixed keys
        assert reduced["clock_cycle:unique_1"] == 12346
        assert reduced["clock_cycle:unique_2"] == 12347

    def test_unique_reduction_with_other_metrics(self):
        """Test unique reduction alongside other metric types."""
        results = [
            self.create_forward_backward_output({
                "clock_cycle:unique": 100,
                "loss:mean": 0.5,
                "accuracy:max": 0.8,
            }),
            self.create_forward_backward_output({
                "clock_cycle:unique": 101,
                "loss:mean": 0.6,
                "accuracy:max": 0.9,
            }),
        ]

        reduced = _metrics_reduction(results)

        # Unique metric behavior
        assert reduced["clock_cycle:unique"] == 100
        assert reduced["clock_cycle:unique_1"] == 101

        # Other metrics should work as normal
        assert reduced["loss:mean"] == 0.55
        assert reduced["accuracy:max"] == 0.9

    def test_unique_reduction_empty_results(self):
        """Test unique reduction with empty results."""
        results = []
        reduced = _metrics_reduction(results)
        assert reduced == {}

    def test_unique_reduction_missing_metric(self):
        """Test unique reduction when some results don't have the metric."""
        results = [
            self.create_forward_backward_output({"clock_cycle:unique": 100}),
            self.create_forward_backward_output({"other_metric:mean": 0.5}),  # Missing clock_cycle
        ]

        reduced = _metrics_reduction(results)

        # Neither metric should be present since they're not in all results
        # This matches the actual behavior of _metrics_reduction which requires
        # all results to have a metric for it to be processed
        assert "clock_cycle:unique" not in reduced
        assert "other_metric:mean" not in reduced
        assert len(reduced) == 0   # Should be empty

    def test_unique_reduction_with_float_values(self):
        """Test unique reduction with float values."""
        results = [
            self.create_forward_backward_output({"timestamp:unique": 1234567890.123}),
            self.create_forward_backward_output({"timestamp:unique": 1234567890.456}),
        ]

        reduced = _metrics_reduction(results)

        assert reduced["timestamp:unique"] == 1234567890.123
        assert reduced["timestamp:unique_1"] == 1234567890.456

    def test_invalid_reduction_type_raises_assertion(self):
        """Test that invalid reduction types raise AssertionError."""
        results = [
            self.create_forward_backward_output({"invalid:nonexistent": 100})
        ]

        with pytest.raises(AssertionError, match="Invalid reduction nonexistent"):
            _metrics_reduction(results)
