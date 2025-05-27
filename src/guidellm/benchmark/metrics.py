from guidellm.objects import (
    StandardBaseModel,
    StatusDistributionSummary,
)

from pydantic import Field

__all__ = [
    "BenchmarkMetrics",
    "GenerativeMetrics",
]


class BenchmarkMetrics(StandardBaseModel):
    """
    A serializable model representing the metrics for a benchmark run.
    """

    requests_per_second: StatusDistributionSummary = Field(
        description="The distribution of requests per second for the benchmark.",
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="The distribution of requests concurrency for the benchmark.",
    )


class GenerativeMetrics(BenchmarkMetrics):
    """
    A serializable model representing the metrics for a generative benchmark run.
    """

    request_latency: StatusDistributionSummary = Field(
        description="The distribution of latencies for the completed requests.",
    )
    prompt_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the prompts for completed, "
            "errored, and all requests."
        )
    )
    output_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the outputs for completed, "
            "errored, and all requests."
        )
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies to receiving the first token in "
            "milliseconds for completed, errored, and all requests."
        ),
    )
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies per output token in milliseconds for "
            "completed, errored, and all requests. "
            "This includes the time to generate the first token and all other tokens."
        ),
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies between tokens in milliseconds for "
            "completed, errored, and all requests."
        ),
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of output tokens per second for completed, "
            "errored, and all requests."
        ),
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of tokens per second, including prompt and output tokens "
            "for completed, errored, and all requests."
        ),
    )
