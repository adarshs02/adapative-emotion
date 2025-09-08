"""
EmoBIRDv2 Pipeline API

Provides a reusable interface to run the multi-step EmoBIRDv2 pipeline.

Public API:
- EmoBIRDConfig: configuration dataclass for model, tokens, attempts, and logging
- EmoBIRDPipeline: class exposing step_* methods and run_for_text()
- run_pipeline_for_text: functional wrapper for one-shot usage

Example:
    from EmoBIRDv2.pipeline import EmoBIRDConfig, EmoBIRDPipeline, run_pipeline_for_text

    # Functional
    result = run_pipeline_for_text(
        situation="I argued with a close friend and now I feel conflicted.",
        config=EmoBIRDConfig(
            model="meta-llama/llama-3.1-8b-instruct",
            attempts=4,
            log_raw=False,
        ),
    )

    # Class-based
    pipe = EmoBIRDPipeline(EmoBIRDConfig(model="meta-llama/llama-3.1-8b-instruct"))
    result = pipe.run_for_text(situation="Work has been very stressful lately.")
"""

from .core import EmoBIRDConfig, EmoBIRDPipeline, run_pipeline_for_text

__all__ = [
    "EmoBIRDConfig",
    "EmoBIRDPipeline",
    "run_pipeline_for_text",
]
