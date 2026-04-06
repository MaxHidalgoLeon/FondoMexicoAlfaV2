import unittest

from src.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_runs_without_errors(self) -> None:
        results = run_pipeline()
        self.assertIn("backtest", results)
        self.assertGreater(results["summary"]["universe_size"], 0)
        self.assertGreaterEqual(results["summary"]["metrics"]["annualized_vol"], 0.0)


if __name__ == "__main__":
    unittest.main()
