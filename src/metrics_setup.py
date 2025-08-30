import os, time
from contextlib import contextmanager
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

def init_metrics(service_name: str):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    exporter = None
    if endpoint:
        exporter = OTLPMetricExporter(endpoint=endpoint, headers=dict(h.split("=",1) for h in headers.split(","))
            if headers else None)
    reader = PeriodicExportingMetricReader(exporter) if exporter else None

    provider = MeterProvider(metric_readers=[reader] if reader else [])
    set_meter_provider(provider)
    meter = provider.get_meter(service_name)

    # core metrics
    rag_latency_ms = meter.create_histogram("rag_latency_ms", unit="ms", description="End-to-end RAG latency")
    retrieval_latency_ms = meter.create_histogram("retrieval_latency_ms", unit="ms", description="Retriever latency")
    end_to_end_latency_ms = meter.create_histogram("end_to_end_latency_ms", unit="ms", description="Total latency to generate a response")
    llm_latency_ms = meter.create_histogram("llm_latency_ms", unit="ms", description="LLM call latency")
    requests_total = meter.create_counter("requests_total", description="RAG requests processed")
    oos_total = meter.create_counter("out_of_scope_total", description="Out-of-scope requests")
    tokens_in = meter.create_histogram("llm_tokens_in", description="Prompt tokens")
    tokens_out = meter.create_histogram("llm_tokens_out", description="Completion tokens")

    return {
        "rag_latency_ms": rag_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "llm_latency_ms": llm_latency_ms,
        "requests_total": requests_total,
        "out_of_scope_total": oos_total,
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
    }

@contextmanager
def time_histogram(hist, **attrs):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        hist.record(dt_ms, attributes=attrs)
