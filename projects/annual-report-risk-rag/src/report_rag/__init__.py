"""Annual-report risk-metric extraction with hybrid retrieval and grounded LLM extraction.

Public entry points:

* :class:`report_rag.pipeline.RiskExtractionPipeline` - end-to-end ingest -> retrieve -> extract.
* :mod:`report_rag.evaluation` - scoring against gold labels.
* :mod:`report_rag.cli` - the ``report-rag`` command line interface.
"""

from report_rag.pipeline import RiskExtractionPipeline
from report_rag.schemas import ExtractedMetric, ExtractionReport, MetricSpec

__all__ = ["ExtractedMetric", "ExtractionReport", "MetricSpec", "RiskExtractionPipeline"]
__version__ = "0.1.0"
