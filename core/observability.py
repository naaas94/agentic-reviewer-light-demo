"""
Observability and metrics collection for Agentic Reviewer.

Provides comprehensive runtime metrics, system monitoring, and event logging.
"""

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PhaseMetrics:
    """Metrics for a single execution phase."""
    phase_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    samples_processed: int = 0
    errors: List[str] = field(default_factory=list)
    
    def finish(self, samples: int = 0):
        """Mark phase as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.samples_processed = samples


@dataclass
class LatencyDistribution:
    """Latency distribution statistics."""
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    stddev_ms: float


@dataclass
class SystemMetrics:
    """System resource usage during run."""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class ObservabilityCollector:
    """Collects runtime observability data."""
    
    def __init__(self, run_id: str):
        """Initialize observability collector.
        
        Args:
            run_id: Unique identifier for this run
        """
        self.run_id = run_id
        self.phases: Dict[str, PhaseMetrics] = {}
        self.latencies: List[float] = []
        self.retry_counts: List[int] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.cache_operations: List[Dict[str, Any]] = []
        self.system_snapshots: List[SystemMetrics] = []
        self.events: List[Dict[str, Any]] = []
        self._start_time = time.time()
        
    def start_phase(self, phase_name: str):
        """Start tracking a phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'data_generation', 'llm_review')
        """
        self.phases[phase_name] = PhaseMetrics(
            phase_name=phase_name,
            start_time=time.time()
        )
        self.log_event("phase_start", {"phase": phase_name})
        
    def end_phase(self, phase_name: str, samples: int = 0):
        """End tracking a phase.
        
        Args:
            phase_name: Name of the phase
            samples: Number of samples processed in this phase
        """
        if phase_name in self.phases:
            self.phases[phase_name].finish(samples)
            phase = self.phases[phase_name]
            self.log_event("phase_end", {
                "phase": phase_name,
                "duration_ms": phase.duration_ms,
                "samples": samples
            })
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latencies.append(latency_ms)
        
    def record_retry(self, sample_id: str, attempt: int, error_type: Optional[str] = None):
        """Record a retry attempt.
        
        Args:
            sample_id: Identifier for the sample
            attempt: Retry attempt number (0-indexed)
            error_type: Optional error type that triggered the retry
        """
        self.retry_counts.append(attempt)
        self.log_event("retry", {
            "sample_id": sample_id,
            "attempt": attempt,
            "error_type": error_type
        })
        
    def record_error(self, error_type: str, message: str, sample_id: Optional[str] = None):
        """Record an error.
        
        Args:
            error_type: Category of error (e.g., 'timeout', 'validation', 'network')
            message: Error message
            sample_id: Optional sample identifier if error is sample-specific
        """
        self.error_counts[error_type] += 1
        self.log_event("error", {
            "type": error_type,
            "message": message,
            "sample_id": sample_id
        })
        
    def record_cache_operation(self, operation: str, hit: bool, latency_ms: float = 0.0):
        """Record cache operation.
        
        Args:
            operation: Operation type ('get', 'set', 'miss')
            hit: Whether it was a cache hit
            latency_ms: Latency of the operation
        """
        self.cache_operations.append({
            "operation": operation,
            "hit": hit,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })
        
    def capture_system_snapshot(self):
        """Capture current system resource usage."""
        try:
            import psutil
            process = psutil.Process()
            cpu = process.cpu_percent(interval=0.1)
            memory = process.memory_info()
            
            snapshot = SystemMetrics(
                cpu_percent=cpu,
                memory_mb=memory.rss / (1024 * 1024),
                memory_percent=process.memory_percent(),
                timestamp=time.time()
            )
            
            # Try to get GPU metrics if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    snapshot.gpu_utilization = gpu.load * 100
                    snapshot.gpu_memory_mb = gpu.memoryUsed
            except (ImportError, Exception):
                # GPUtil not available or no GPU
                pass
                
            self.system_snapshots.append(snapshot)
        except ImportError:
            # psutil not available - skip system metrics
            pass
        except Exception:
            # System metrics collection failed - continue without them
            pass
        
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured event.
        
        Args:
            event_type: Type of event (e.g., 'phase_start', 'error', 'retry')
            data: Event-specific data
        """
        self.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })
        
    def get_latency_distribution(self) -> Optional[LatencyDistribution]:
        """Calculate latency distribution statistics.
        
        Returns:
            LatencyDistribution if latencies exist, None otherwise
        """
        if not self.latencies:
            return None
            
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return LatencyDistribution(
            count=n,
            min_ms=min(self.latencies),
            max_ms=max(self.latencies),
            mean_ms=statistics.mean(self.latencies),
            median_ms=statistics.median(self.latencies),
            p50_ms=sorted_latencies[int(n * 0.50)],
            p95_ms=sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1],
            stddev_ms=statistics.stdev(self.latencies) if n > 1 else 0.0
        )
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary.
        
        Returns:
            Dictionary containing all observability metrics
        """
        latency_dist = self.get_latency_distribution()
        
        # Calculate total duration
        total_duration = time.time() - self._start_time
        
        # Calculate throughput from review phase
        review_phase = self.phases.get("llm_review")
        if review_phase and review_phase.duration_ms:
            review_duration_s = review_phase.duration_ms / 1000
            total_samples = review_phase.samples_processed
            throughput = total_samples / review_duration_s if review_duration_s > 0 else 0
        else:
            throughput = 0
            total_samples = sum(p.samples_processed for p in self.phases.values())
        
        # Average retries
        avg_retries = statistics.mean(self.retry_counts) if self.retry_counts else 0
        
        # System metrics summary
        if self.system_snapshots:
            avg_cpu = statistics.mean([s.cpu_percent for s in self.system_snapshots])
            avg_memory_mb = statistics.mean([s.memory_mb for s in self.system_snapshots])
            peak_memory_mb = max([s.memory_mb for s in self.system_snapshots])
            avg_memory_percent = statistics.mean([s.memory_percent for s in self.system_snapshots])
        else:
            avg_cpu = 0
            avg_memory_mb = 0
            peak_memory_mb = 0
            avg_memory_percent = 0
        
        # Cache statistics
        cache_hits = sum(1 for op in self.cache_operations if op["hit"])
        cache_misses = sum(1 for op in self.cache_operations if not op["hit"])
        cache_hit_rate = (cache_hits / len(self.cache_operations) * 100) if self.cache_operations else 0
        
        return {
            "run_id": self.run_id,
            "total_duration_s": total_duration,
            "phases": {
                name: {
                    "duration_ms": p.duration_ms,
                    "samples_processed": p.samples_processed,
                    "errors": len(p.errors),
                    "start_time": p.start_time,
                    "end_time": p.end_time
                }
                for name, p in self.phases.items()
            },
            "latency": {
                "distribution": asdict(latency_dist) if latency_dist else None,
                "total_samples": len(self.latencies)
            },
            "throughput": {
                "samples_per_second": throughput,
                "total_samples": total_samples
            },
            "retries": {
                "total_retries": sum(self.retry_counts),
                "samples_with_retries": len(self.retry_counts),
                "average_retries_per_sample": avg_retries,
                "max_retries": max(self.retry_counts) if self.retry_counts else 0
            },
            "errors": dict(self.error_counts),
            "system": {
                "average_cpu_percent": round(avg_cpu, 2),
                "average_memory_mb": round(avg_memory_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "average_memory_percent": round(avg_memory_percent, 2),
                "snapshots_count": len(self.system_snapshots)
            },
            "cache": {
                "total_operations": len(self.cache_operations),
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate_percent": round(cache_hit_rate, 2)
            },
            "events_count": len(self.events)
        }
        
    def save_to_file(self, filepath: str):
        """Save observability data to JSON file.
        
        Args:
            filepath: Path to save the observability JSON file
        """
        data = {
            "summary": self.get_summary(),
            "events": self.events[-1000:],  # Last 1000 events to avoid huge files
            "system_snapshots": [
                asdict(s) for s in self.system_snapshots
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def save_event_log(self, filepath: str):
        """Save structured event log (one JSON line per event).
        
        Args:
            filepath: Path to save the event log file (JSONL format)
        """
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')

