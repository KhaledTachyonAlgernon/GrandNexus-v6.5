"""
monitor.py - Cognitive Process Monitoring Module for GrandNexus

This module provides real-time monitoring and analytics of GrandNexus's cognitive processes.
It acts as an awareness layer, observing the behavior and performance of all cognitive
components, collecting metrics, detecting anomalies, and providing insights
that can inform metacognitive decisions.

Key features:
- Real-time monitoring of all cognitive module activities
- Collection and aggregation of performance metrics
- Detection of processing bottlenecks and anomalies
- Resource utilization tracking
- Cognitive load estimation
- Integration with adaptive mechanisms for self-regulation

The design emphasizes:
- Minimal interference with monitored processes
- Extensible metrics collection framework
- Configurable thresholds and monitoring policies
- Integration with other metacognitive components
"""

import sys  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook
import logging
import threading
import time
import uuid
import psutil
import os
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import numpy as np
import datetime

# Assuming these exist in GrandNexus architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.metacognition.confidence import ConfidenceEvaluator

# Metric types for categorization
class MetricType(Enum):
    """Types of metrics that can be monitored"""
    PERFORMANCE = auto()      # Processing speed, throughput, latency
    RESOURCE = auto()         # Memory usage, CPU usage, etc.
    ACTIVITY = auto()         # Module operation counts, call frequency
    QUALITY = auto()          # Confidence scores, precision, error rates
    STABILITY = auto()        # Error counts, exception rates, etc.
    COGNITIVE_LOAD = auto()   # Estimated load on various cognitive resources
    EMERGING = auto()         # Metrics that emerge from system interactions

class MetricPriority(Enum):
    """Priority levels for metrics, influencing sampling rate and alerting"""
    CRITICAL = 0   # Essential metrics requiring immediate attention if thresholds crossed
    HIGH = 1       # Important metrics requiring prompt attention
    MEDIUM = 2     # Standard metrics for regular monitoring
    LOW = 3        # Background metrics collected less frequently
    DEBUG = 4      # Metrics collected only during debug sessions

@dataclass
class MetricDefinition:
    """Definition of a metric to be monitored"""
    name: str
    description: str
    type: MetricType
    unit: str = ""
    priority: MetricPriority = MetricPriority.MEDIUM
    sampling_interval: float = 1.0  # seconds
    smoothing_factor: float = 0.2   # For exponential smoothing (0-1)
    alert_threshold_low: Optional[float] = None
    alert_threshold_high: Optional[float] = None
    derived: bool = False  # True if calculated from other metrics

@dataclass
class MetricValue:
    """A single metric measurement with metadata"""
    metric_id: str
    timestamp: float
    value: float
    source_module: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoredSystem:
    """Information about the system being monitored"""
    cpu_count: int
    total_memory: int  # bytes
    start_time: float
    platform: str
    modules: Set[str] = field(default_factory=set)

@dataclass
class MonitorAlert:
    """Alert raised when a metric crosses a threshold"""
    alert_id: str
    metric_id: str
    timestamp: float
    severity: str  # "warning", "error", "critical"
    message: str
    value: float
    threshold: float
    is_active: bool = True

@dataclass
class PerformanceReport:
    """Comprehensive report on system performance"""
    timestamp: float
    time_period: str  # e.g., "last_minute", "last_hour", "last_day"
    metrics_summary: Dict[str, Dict[str, float]]  # metric_id -> {avg, min, max, std}
    bottlenecks: List[Dict[str, Any]]
    resource_usage: Dict[str, float]
    cognitive_load: Dict[str, float]
    alerts: List[MonitorAlert]
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

class CognitiveMonitor:
    """
    Central monitoring system for tracking the performance and behavior
    of GrandNexus's cognitive modules.

    This module acts as an observational layer, collecting metrics,
    detecting anomalies, and providing insights without directly
    intervening in cognitive processes.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 confidence_evaluator: Optional[ConfidenceEvaluator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive monitor.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            confidence_evaluator: Reference to confidence evaluation module
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("GrandNexus.Metacognition.Monitor")
        self.nexus_core = nexus_core
        self.confidence_evaluator = confidence_evaluator

        # Initialize configuration
        self.config = config or {}
        self.monitoring_interval = self.config.get("monitoring_interval", 1.0)  # seconds
        self.history_length = self.config.get("history_length", 3600)  # number of samples to keep
        self.alert_check_interval = self.config.get("alert_check_interval", 5.0)  # seconds
        self.report_interval = self.config.get("report_interval", 300.0)  # seconds

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # System information
        self.system_info = MonitoredSystem(
            cpu_count=os.cpu_count() or 1,
            total_memory=psutil.virtual_memory().total,
            start_time=self.start_time,
            platform=f"Python {sys.version.split()[0]} on {sys.platform}"
        )

        # Metrics registry
        self.metrics: Dict[str, MetricDefinition] = {}

        # Metrics storage
        self.metric_histories: Dict[str, deque[MetricValue]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

        # Current values (latest for each metric)
        self.current_values: Dict[str, MetricValue] = {}

        # Smoothed values (exponential moving average)
        self.smoothed_values: Dict[str, float] = {}

        # Alerts
        self.active_alerts: Dict[str, MonitorAlert] = {}
        self.alert_history: deque[MonitorAlert] = deque(maxlen=1000)

        # Bottleneck detection
        self.bottleneck_thresholds = self.config.get("bottleneck_thresholds", {
            "cpu_usage": 90.0,  # percentage
            "memory_usage": 85.0,  # percentage
            "queue_growth_rate": 0.2,  # items per second
            "response_time": 2.0  # seconds
        })

        # Performance reports
        self.reports: deque[PerformanceReport] = deque(maxlen=100)

        # Monitoring threads
        self.monitor_thread = None
        self.alert_thread = None
        self.report_thread = None
        self.stop_event = threading.Event()

        # Custom metric collectors (module_name -> list of collector functions)
        self.custom_collectors: Dict[str, List[Callable[[], Dict[str, float]]]] = defaultdict(list)

        self.logger.info(f"CognitiveMonitor initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the monitor with standard metrics"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register standard metrics
                self._register_standard_metrics()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Register for callbacks and events
                self._subscribe_to_events()

                self.initialized = True
                self.logger.info("CognitiveMonitor initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"CognitiveMonitor initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the monitoring processes"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start the monitoring thread
                self.stop_event.clear()
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="Monitor_Thread",
                    daemon=True
                )
                self.monitor_thread.start()

                # Start the alert checking thread
                self.alert_thread = threading.Thread(
                    target=self._alert_check_loop,
                    name="Alert_Thread",
                    daemon=True
                )
                self.alert_thread.start()

                # Start the report generation thread
                self.report_thread = threading.Thread(
                    target=self._report_generation_loop,
                    name="Report_Thread",
                    daemon=True
                )
                self.report_thread.start()

                self.running = True
                self.logger.info("CognitiveMonitor started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start CognitiveMonitor: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the monitoring processes"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Signal threads to stop
                self.stop_event.set()

                # Wait for threads to terminate
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=5.0)

                if self.alert_thread and self.alert_thread.is_alive():
                    self.alert_thread.join(timeout=5.0)

                if self.report_thread and self.report_thread.is_alive():
                    self.report_thread.join(timeout=5.0)

                self.running = False
                self.logger.info("CognitiveMonitor stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping CognitiveMonitor: {str(e)}")
                return False

    def register_metric(self, metric: MetricDefinition) -> bool:
        """
        Register a new metric to be monitored.

        Args:
            metric: The metric definition

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            if metric.name in self.metrics:
                self.logger.warning(f"Metric {metric.name} already exists, updating")

            self.metrics[metric.name] = metric
            self.logger.info(f"Registered metric: {metric.name} ({metric.type.name})")
            return True

    def record_metric(self, metric_name: str, value: float,
                     source_module: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            source_module: Name of the module reporting the metric
            context: Optional contextual information

        Returns:
            True if recording succeeded, False otherwise
        """
        if metric_name not in self.metrics:
            self.logger.warning(f"Unknown metric: {metric_name}")
            return False

        try:
            # Create metric value object
            metric_value = MetricValue(
                metric_id=metric_name,
                timestamp=time.time(),
                value=value,
                source_module=source_module,
                context=context or {}
            )

            # Store the value
            with self.lock:
                # Add to history
                self.metric_histories[metric_name].append(metric_value)

                # Update current value
                self.current_values[metric_name] = metric_value

                # Update smoothed value
                smoothing_factor = self.metrics[metric_name].smoothing_factor
                if metric_name in self.smoothed_values:
                    self.smoothed_values[metric_name] = (
                        smoothing_factor * value +
                        (1 - smoothing_factor) * self.smoothed_values[metric_name]
                    )
                else:
                    self.smoothed_values[metric_name] = value

            return True
        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name}: {str(e)}")
            return False

    def get_current_value(self, metric_name: str) -> Optional[float]:
        """
        Get the current value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Current value, or None if the metric doesn't exist
        """
        with self.lock:
            metric_value = self.current_values.get(metric_name)
            return metric_value.value if metric_value else None

    def get_smoothed_value(self, metric_name: str) -> Optional[float]:
        """
        Get the smoothed value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Smoothed value, or None if the metric doesn't exist
        """
        with self.lock:
            return self.smoothed_values.get(metric_name)

    def get_metric_history(self, metric_name: str,
                         time_window: Optional[float] = None) -> List[MetricValue]:
        """
        Get the history of a metric over a time window.

        Args:
            metric_name: Name of the metric
            time_window: Optional time window in seconds (if None, return all history)

        Returns:
            List of metric values within the time window
        """
        with self.lock:
            if metric_name not in self.metric_histories:
                return []

            history = list(self.metric_histories[metric_name])

            if time_window is not None:
                cutoff_time = time.time() - time_window
                history = [v for v in history if v.timestamp >= cutoff_time]

            return history

    def get_active_alerts(self) -> List[MonitorAlert]:
        """Get all currently active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, time_window: Optional[float] = None) -> List[MonitorAlert]:
        """
        Get alert history over a time window.

        Args:
            time_window: Optional time window in seconds (if None, return all history)

        Returns:
            List of alerts within the time window
        """
        with self.lock:
            alerts = list(self.alert_history)

            if time_window is not None:
                cutoff_time = time.time() - time_window
                alerts = [a for a in alerts if a.timestamp >= cutoff_time]

            return alerts

    def get_latest_report(self) -> Optional[PerformanceReport]:
        """Get the most recent performance report"""
        with self.lock:
            return self.reports[-1] if self.reports else None

    def get_reports(self, time_window: Optional[float] = None) -> List[PerformanceReport]:
        """
        Get performance reports over a time window.

        Args:
            time_window: Optional time window in seconds (if None, return all reports)

        Returns:
            List of performance reports within the time window
        """
        with self.lock:
            reports = list(self.reports)

            if time_window is not None:
                cutoff_time = time.time() - time_window
                reports = [r for r in reports if r.timestamp >= cutoff_time]

            return reports

    def register_custom_collector(self, module_name: str,
                                collector_func: Callable[[], Dict[str, float]]) -> bool:
        """
        Register a custom metric collector function for a module.

        Args:
            module_name: Name of the module
            collector_func: Function that returns a dict mapping metric names to values

        Returns:
            True if registration succeeded, False otherwise
        """
        with self.lock:
            self.custom_collectors[module_name].append(collector_func)
            self.logger.info(f"Registered custom collector for module: {module_name}")
            return True

    def trigger_custom_report(self, metrics_filter: Optional[Set[str]] = None,
                            time_window: float = 300.0) -> PerformanceReport:
        """
        Generate a custom performance report on demand.

        Args:
            metrics_filter: Optional set of metric names to include
            time_window: Time window in seconds for metrics analysis

        Returns:
            Generated performance report
        """
        # Collect metrics within time window
        metrics_data = self._collect_metrics_data(time_window, metrics_filter)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(metrics_data)

        # Get resource usage
        resource_usage = self._get_current_resource_usage()

        # Estimate cognitive load
        cognitive_load = self._estimate_cognitive_load()

        # Get active alerts
        alerts = list(self.active_alerts.values())

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics_data, bottlenecks, resource_usage, alerts
        )

        # Create report
        report = PerformanceReport(
            timestamp=time.time(),
            time_period=f"last_{int(time_window)}s",
            metrics_summary=metrics_data,
            bottlenecks=bottlenecks,
            resource_usage=resource_usage,
            cognitive_load=cognitive_load,
            alerts=alerts,
            recommendations=recommendations
        )

        # Store the report
        with self.lock:
            self.reports.append(report)

        return report

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.confidence_evaluator:
                dependencies.append("confidence_evaluator")

            self.nexus_core.register_module(
                name="cognitive_monitor",
                module=self,
                dependencies=dependencies
            )

            # Get all registered modules
            modules = self.nexus_core.list_modules()
            self.system_info.modules.update(modules)

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _subscribe_to_events(self) -> None:
        """Subscribe to events from NexusCore and other modules"""
        if not self.nexus_core:
            return

        # In a real implementation, this would set up event subscriptions
        # For now, it's just a placeholder
        self.logger.info("Event subscriptions set up")

    def _register_standard_metrics(self) -> None:
        """Register standard system and cognitive metrics"""
        # System resource metrics
        self.register_metric(MetricDefinition(
            name="system.cpu_usage",
            description="Overall CPU usage percentage",
            type=MetricType.RESOURCE,
            unit="%",
            priority=MetricPriority.HIGH,
            alert_threshold_high=90.0
        ))

        self.register_metric(MetricDefinition(
            name="system.memory_usage",
            description="Overall memory usage percentage",
            type=MetricType.RESOURCE,
            unit="%",
            priority=MetricPriority.HIGH,
            alert_threshold_high=85.0
        ))

        self.register_metric(MetricDefinition(
            name="system.memory_available",
            description="Available system memory",
            type=MetricType.RESOURCE,
            unit="MB",
            priority=MetricPriority.MEDIUM,
            alert_threshold_low=500.0
        ))

        # Core metrics
        self.register_metric(MetricDefinition(
            name="core.message_queue_size",
            description="Size of the NexusCore message queue",
            type=MetricType.ACTIVITY,
            unit="messages",
            priority=MetricPriority.HIGH,
            alert_threshold_high=100
        ))

        self.register_metric(MetricDefinition(
            name="core.messages_per_second",
            description="Rate of message processing",
            type=MetricType.PERFORMANCE,
            unit="msg/s",
            priority=MetricPriority.MEDIUM
        ))

        self.register_metric(MetricDefinition(
            name="core.active_modules",
            description="Number of active modules",
            type=MetricType.ACTIVITY,
            unit="modules",
            priority=MetricPriority.MEDIUM
        ))

        # Performance metrics
        self.register_metric(MetricDefinition(
            name="performance.response_time",
            description="Average response time for requests",
            type=MetricType.PERFORMANCE,
            unit="ms",
            priority=MetricPriority.HIGH,
            alert_threshold_high=5000.0
        ))

        self.register_metric(MetricDefinition(
            name="performance.throughput",
            description="Overall system throughput",
            type=MetricType.PERFORMANCE,
            unit="ops/s",
            priority=MetricPriority.MEDIUM
        ))

        # Cognitive metrics
        self.register_metric(MetricDefinition(
            name="cognitive.working_memory_load",
            description="Load on working memory",
            type=MetricType.COGNITIVE_LOAD,
            unit="%",
            priority=MetricPriority.HIGH,
            derived=True,
            alert_threshold_high=90.0
        ))

        self.register_metric(MetricDefinition(
            name="cognitive.semantic_memory_access_rate",
            description="Rate of semantic memory accesses",
            type=MetricType.ACTIVITY,
            unit="accesses/s",
            priority=MetricPriority.MEDIUM
        ))

        self.register_metric(MetricDefinition(
            name="cognitive.reasoning_confidence",
            description="Average confidence in reasoning results",
            type=MetricType.QUALITY,
            unit="%",
            priority=MetricPriority.HIGH,
            alert_threshold_low=70.0
        ))

        self.register_metric(MetricDefinition(
            name="cognitive.learning_rate",
            description="Rate of new knowledge acquisition",
            type=MetricType.PERFORMANCE,
            unit="items/min",
            priority=MetricPriority.MEDIUM
        ))

        # Stability metrics
        self.register_metric(MetricDefinition(
            name="stability.error_rate",
            description="Rate of errors across all modules",
            type=MetricType.STABILITY,
            unit="errors/min",
            priority=MetricPriority.CRITICAL,
            alert_threshold_high=5.0
        ))

        self.register_metric(MetricDefinition(
            name="stability.module_restarts",
            description="Number of module restarts",
            type=MetricType.STABILITY,
            unit="restarts",
            priority=MetricPriority.HIGH,
            alert_threshold_high=3.0
        ))

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects metrics at regular intervals"""
        self.logger.info("Monitoring loop started")

        last_collection_times = defaultdict(float)

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                # Collect system metrics
                self._collect_system_metrics()

                # Collect metrics from each module based on sampling interval
                for metric_name, metric_def in self.metrics.items():
                    if current_time - last_collection_times[metric_name] >= metric_def.sampling_interval:
                        self._collect_metric(metric_name, metric_def)
                        last_collection_times[metric_name] = current_time

                # Collect metrics from custom collectors
                self._run_custom_collectors()

                # Collect derived metrics
                self._calculate_derived_metrics()

                # Sleep until next collection cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error

    def _alert_check_loop(self) -> None:
        """Loop that checks for metrics crossing alert thresholds"""
        self.logger.info("Alert check loop started")

        while not self.stop_event.is_set():
            try:
                # Check all metrics against their thresholds
                self._check_alert_thresholds()

                # Sleep until next check
                time.sleep(self.alert_check_interval)

            except Exception as e:
                self.logger.error(f"Error in alert check loop: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error

    def _report_generation_loop(self) -> None:
        """Loop that generates periodic performance reports"""
        self.logger.info("Report generation loop started")

        next_report_time = time.time() + self.report_interval

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                if current_time >= next_report_time:
                    # Generate a standard report
                    self._generate_periodic_report()
                    next_report_time = current_time + self.report_interval

                # Sleep for a bit
                time.sleep(min(30.0, next_report_time - current_time))

            except Exception as e:
                self.logger.error(f"Error in report generation loop: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error
                next_report_time = time.time() + self.report_interval

    def _collect_system_metrics(self) -> None:
        """Collect basic system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("system.cpu_usage", cpu_percent, "system")

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)

            self.record_metric("system.memory_usage", memory_percent, "system")
            self.record_metric("system.memory_available", memory_available_mb, "system")

            # Collect NexusCore metrics if available
            if self.nexus_core:
                try:
                    core_status = self.nexus_core.get_status()

                    # Message queue size
                    if "message_queue_size" in core_status:
                        self.record_metric(
                            "core.message_queue_size",
                            core_status["message_queue_size"],
                            "nexus_core"
                        )

                    # Active modules
                    if "active_modules" in core_status:
                        self.record_metric(
                            "core.active_modules",
                            len(core_status["active_modules"]),
                            "nexus_core"
                        )

                except Exception as e:
                    self.logger.warning(f"Error collecting NexusCore metrics: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")

    def _collect_metric(self, metric_name: str, metric_def: MetricDefinition) -> None:
        """
        Collect a specific metric based on its definition.
        This would typically dispatch to specialized collection methods.
        """
        # Most metrics are collected directly from modules
        # This method handles special cases or polling metrics
        pass

    def _run_custom_collectors(self) -> None:
        """Run all registered custom metric collectors"""
        for module_name, collectors in self.custom_collectors.items():
            for collector in collectors:
                try:
                    metrics = collector()
                    for metric_name, value in metrics.items():
                        if metric_name in self.metrics:
                            self.record_metric(metric_name, value, module_name)
                        else:
                            self.logger.warning(f"Unknown metric from collector: {metric_name}")

                except Exception as e:
                    self.logger.error(f"Error in custom collector for {module_name}: {str(e)}")

    def _calculate_derived_metrics(self) -> None:
        """Calculate metrics that are derived from other metrics"""
        # Example: Calculate messages per second from queue size changes
        try:
            # Working memory load
            # (This would typically use actual working memory metrics)
            # Here we'll just use a placeholder calculation
            if "system.memory_usage" in self.current_values:
                memory_usage = self.current_values["system.memory_usage"].value
                # Simplified model relating system memory to cognitive load
                working_memory_load = min(100.0, memory_usage * 1.2)
                self.record_metric("cognitive.working_memory_load", working_memory_load, "monitor")

            # Message processing rate
            message_queue_history = self.get_metric_history("core.message_queue_size", 10.0)
            if len(message_queue_history) >= 2:
                oldest = message_queue_history[0]
                newest = message_queue_history[-1]
                time_diff = newest.timestamp - oldest.timestamp
                if time_diff > 0:
                    queue_diff = oldest.value - newest.value  # Processed = decreased queue
                    if queue_diff > 0:  # Only if queue decreased
                        rate = queue_diff / time_diff
                        self.record_metric("core.messages_per_second", rate, "monitor")

        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {str(e)}")

    def _check_alert_thresholds(self) -> None:
        """Check all metrics against their alert thresholds"""
        try:
            # Check each metric with defined thresholds
            for metric_name, metric_def in self.metrics.items():
                current_value = self.get_current_value(metric_name)
                if current_value is None:
                    continue

                # Check high threshold
                if (metric_def.alert_threshold_high is not None and
                    current_value > metric_def.alert_threshold_high):
                    self._trigger_alert(
                        metric_name,
                        current_value,
                        metric_def.alert_threshold_high,
                        "high",
                        f"{metric_name} exceeded high threshold: {current_value} > {metric_def.alert_threshold_high} {metric_def.unit}"
                    )
                else:
                    # Check if there's an active high alert to resolve
                    self._resolve_alert(metric_name, "high")

                # Check low threshold
                if (metric_def.alert_threshold_low is not None and
                    current_value < metric_def.alert_threshold_low):
                    self._trigger_alert(
                        metric_name,
                        current_value,
                        metric_def.alert_threshold_low,
                        "low",
                        f"{metric_name} below low threshold: {current_value} < {metric_def.alert_threshold_low} {metric_def.unit}"
                    )
                else:
                    # Check if there's an active low alert to resolve
                    self._resolve_alert(metric_name, "low")

        except Exception as e:
            self.logger.error(f"Error checking alert thresholds: {str(e)}")

    def _trigger_alert(self, metric_name: str, current_value: float,
                      threshold: float, threshold_type: str, message: str) -> None:
        """
        Trigger an alert for a metric crossing a threshold.

        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            threshold: Threshold that was crossed
            threshold_type: Type of threshold ('high' or 'low')
            message: Alert message
        """
        # Determine severity based on metric priority
        metric_def = self.metrics.get(metric_name)
        if not metric_def:
            return

        severity_map = {
            MetricPriority.CRITICAL: "critical",
            MetricPriority.HIGH: "error",
            MetricPriority.MEDIUM: "warning",
            MetricPriority.LOW: "info",
            MetricPriority.DEBUG: "debug"
        }
        severity = severity_map.get(metric_def.priority, "warning")

        # Create alert ID
        alert_id = f"{metric_name}_{threshold_type}_{int(time.time())}"

        # Check if this alert is already active
        existing_alert_id = f"{metric_name}_{threshold_type}"
        if existing_alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[existing_alert_id]
            alert.timestamp = time.time()
            alert.value = current_value
            alert.message = message

            # Log the continuation of the alert
            self.logger.warning(f"Alert continuing: {message}")
            return

        # Create new alert
        alert = MonitorAlert(
            alert_id=alert_id,
            metric_id=metric_name,
            timestamp=time.time(),
            severity=severity,
            message=message,
            value=current_value,
            threshold=threshold,
            is_active=True
        )

        # Store the alert
        with self.lock:
            self.active_alerts[existing_alert_id] = alert
            self.alert_history.append(alert)

        # Log the alert
        log_method = getattr(self.logger, severity, self.logger.warning)
        log_method(f"Alert triggered: {message}")

        # Notify subscribers
        self._notify_alert(alert)

    def _resolve_alert(self, metric_name: str, threshold_type: str) -> None:
        """
        Resolve an active alert when the metric returns to normal range.

        Args:
            metric_name: Name of the metric
            threshold_type: Type of threshold ('high' or 'low')
        """
        # Check if this alert is active
        existing_alert_id = f"{metric_name}_{threshold_type}"
        if existing_alert_id not in self.active_alerts:
            return

        # Get the alert
        alert = self.active_alerts[existing_alert_id]

        # Mark as resolved
        alert.is_active = False

        # Remove from active alerts
        with self.lock:
            del self.active_alerts[existing_alert_id]

        # Log the resolution
        self.logger.info(f"Alert resolved: {alert.message}")

        # Notify subscribers
        self._notify_alert_resolution(alert)

    def _notify_alert(self, alert: MonitorAlert) -> None:
        """
        Notify subscribers about a new alert.

        Args:
            alert: The alert that was triggered
        """
        # In a real implementation, this would notify subscribers
        # Here it's just a placeholder
        if self.nexus_core:
            try:
                # Example: Send a message through NexusCore
                self.nexus_core.send_message(
                    source="cognitive_monitor",
                    target="*",  # Broadcast to all modules
                    message_type="alert",
                    content={
                        "alert_id": alert.alert_id,
                        "metric_id": alert.metric_id,
                        "timestamp": alert.timestamp,
                        "severity": alert.severity,
                        "message": alert.message,
                        "value": alert.value,
                        "threshold": alert.threshold
                    }
                )
            except Exception as e:
                self.logger.error(f"Error notifying alert: {str(e)}")

    def _notify_alert_resolution(self, alert: MonitorAlert) -> None:
        """
        Notify subscribers about an alert resolution.

        Args:
            alert: The alert that was resolved
        """
        # In a real implementation, this would notify subscribers
        # Here it's just a placeholder
        if self.nexus_core:
            try:
                # Example: Send a message through NexusCore
                self.nexus_core.send_message(
                    source="cognitive_monitor",
                    target="*",  # Broadcast to all modules
                    message_type="alert_resolution",
                    content={
                        "alert_id": alert.alert_id,
                        "metric_id": alert.metric_id,
                        "timestamp": time.time(),
                        "message": f"Resolved: {alert.message}"
                    }
                )
            except Exception as e:
                self.logger.error(f"Error notifying alert resolution: {str(e)}")

    def _generate_periodic_report(self) -> None:
        """Generate a standard periodic performance report"""
        try:
            # Generate report for the last reporting interval
            report = self.trigger_custom_report(time_window=self.report_interval)

            # Log report generation
            self.logger.info(f"Generated periodic performance report at {datetime.datetime.now()}")

            # Notify subscribers
            self._notify_report(report)

        except Exception as e:
            self.logger.error(f"Error generating periodic report: {str(e)}")

    def _notify_report(self, report: PerformanceReport) -> None:
        """
        Notify subscribers about a new performance report.

        Args:
            report: The performance report
        """
        # In a real implementation, this would notify subscribers
        # Here it's just a placeholder
        if self.nexus_core:
            try:
                # Example: Send a message through NexusCore
                self.nexus_core.send_message(
                    source="cognitive_monitor",
                    target="metacognition",  # Send to metacognition modules
                    message_type="performance_report",
                    content={
                        "timestamp": report.timestamp,
                        "time_period": report.time_period,
                        "bottlenecks": report.bottlenecks,
                        "resource_usage": report.resource_usage,
                        "cognitive_load": report.cognitive_load,
                        "active_alerts": len(report.alerts),
                        "recommendations": report.recommendations
                    }
                )
            except Exception as e:
                self.logger.error(f"Error notifying report: {str(e)}")

    def _collect_metrics_data(self, time_window: float,
                             metrics_filter: Optional[Set[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Collect metrics data for a report.

        Args:
            time_window: Time window in seconds
            metrics_filter: Optional set of metric names to include

        Returns:
            Dictionary mapping metric names to summary statistics
        """
        result = {}

        # Determine which metrics to include
        metrics_to_include = set(self.metrics.keys())
        if metrics_filter:
            metrics_to_include = metrics_to_include.intersection(metrics_filter)

        # For each metric, collect summary statistics
        for metric_name in metrics_to_include:
            history = self.get_metric_history(metric_name, time_window)

            if not history:
                continue

            values = [entry.value for entry in history]

            # Calculate summary statistics
            summary = {
                "avg": statistics.mean(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "current": values[-1] if values else 0.0
            }

            # Calculate standard deviation if there are enough values
            if len(values) > 1:
                summary["std"] = statistics.stdev(values)
            else:
                summary["std"] = 0.0

            # Add to result
            result[metric_name] = summary

        return result

    def _detect_bottlenecks(self, metrics_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks based on metrics.

        Args:
            metrics_data: Dictionary of metrics summary data

        Returns:
            List of detected bottlenecks with details
        """
        bottlenecks = []

        # Check CPU usage
        if "system.cpu_usage" in metrics_data:
            cpu_usage = metrics_data["system.cpu_usage"]["current"]
            if cpu_usage > self.bottleneck_thresholds.get("cpu_usage", 90.0):
                bottlenecks.append({
                    "type": "resource",
                    "resource": "cpu",
                    "severity": "high" if cpu_usage > 95.0 else "medium",
                    "value": cpu_usage,
                    "threshold": self.bottleneck_thresholds.get("cpu_usage", 90.0),
                    "description": f"CPU usage is {cpu_usage:.1f}%, indicating potential CPU bottleneck"
                })

        # Check memory usage
        if "system.memory_usage" in metrics_data:
            memory_usage = metrics_data["system.memory_usage"]["current"]
            if memory_usage > self.bottleneck_thresholds.get("memory_usage", 85.0):
                bottlenecks.append({
                    "type": "resource",
                    "resource": "memory",
                    "severity": "high" if memory_usage > 95.0 else "medium",
                    "value": memory_usage,
                    "threshold": self.bottleneck_thresholds.get("memory_usage", 85.0),
                    "description": f"Memory usage is {memory_usage:.1f}%, indicating potential memory bottleneck"
                })

        # Check message queue growth
        if "core.message_queue_size" in metrics_data:
            queue_size = metrics_data["core.message_queue_size"]["current"]
            queue_avg = metrics_data["core.message_queue_size"]["avg"]

            if queue_size > 50 and queue_size > queue_avg * 1.5:
                bottlenecks.append({
                    "type": "processing",
                    "resource": "message_queue",
                    "severity": "medium",
                    "value": queue_size,
                    "threshold": queue_avg * 1.5,
                    "description": f"Message queue size ({queue_size}) is growing, indicating processing bottleneck"
                })

        # Check response time
        if "performance.response_time" in metrics_data:
            response_time = metrics_data["performance.response_time"]["current"]
            if response_time > self.bottleneck_thresholds.get("response_time", 2.0):
                bottlenecks.append({
                    "type": "performance",
                    "resource": "response_time",
                    "severity": "high" if response_time > 5.0 else "medium",
                    "value": response_time,
                    "threshold": self.bottleneck_thresholds.get("response_time", 2.0),
                    "description": f"Response time is {response_time:.2f}s, indicating performance bottleneck"
                })

        # Check working memory load
        if "cognitive.working_memory_load" in metrics_data:
            memory_load = metrics_data["cognitive.working_memory_load"]["current"]
            if memory_load > 85.0:
                bottlenecks.append({
                    "type": "cognitive",
                    "resource": "working_memory",
                    "severity": "high" if memory_load > 95.0 else "medium",
                    "value": memory_load,
                    "threshold": 85.0,
                    "description": f"Working memory load is {memory_load:.1f}%, indicating cognitive bottleneck"
                })

        return bottlenecks

    def _get_current_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage for various system resources.

        Returns:
            Dictionary mapping resource names to usage values
        """
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')

            result = {
                "cpu": cpu_percent,
                "memory": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024)
            }

            # Core resources if available
            if self.nexus_core:
                try:
                    core_status = self.nexus_core.get_status()

                    # Add relevant core metrics
                    if "message_queue_size" in core_status:
                        result["message_queue"] = core_status["message_queue_size"]

                    if "active_modules" in core_status:
                        result["active_modules"] = len(core_status["active_modules"])

                except Exception as e:
                    self.logger.warning(f"Error getting NexusCore status: {str(e)}")

            return result

        except Exception as e:
            self.logger.error(f"Error getting resource usage: {str(e)}")
            return {"error": 1.0}

    def _estimate_cognitive_load(self) -> Dict[str, float]:
        """
        Estimate cognitive load across different cognitive resources.

        Returns:
            Dictionary mapping cognitive resource names to load values
        """
        result = {}

        # Working memory load
        working_memory_load = self.get_current_value("cognitive.working_memory_load")
        if working_memory_load is not None:
            result["working_memory"] = working_memory_load

        # Processing load (based on CPU)
        cpu_usage = self.get_current_value("system.cpu_usage")
        if cpu_usage is not None:
            result["processing"] = min(100.0, cpu_usage * 1.1)  # Slightly higher than CPU

        # Attention load (based on number of active modules and tasks)
        if self.nexus_core:
            try:
                core_status = self.nexus_core.get_status()
                active_modules = len(core_status.get("active_modules", []))

                # Estimate attention load based on active modules
                # This is a simplified model
                attention_load = min(100.0, active_modules * 10.0)  # 10% per active module
                result["attention"] = attention_load

            except Exception:
                pass

        # Learning load (based on learning metrics if available)
        learning_rate = self.get_current_value("cognitive.learning_rate")
        if learning_rate is not None:
            # Simplified model: higher learning rate = higher load
            result["learning"] = min(100.0, learning_rate * 5.0)  # 5% per item/min

        # Overall cognitive load (weighted average)
        if result:
            weights = {
                "working_memory": 0.4,
                "processing": 0.3,
                "attention": 0.2,
                "learning": 0.1
            }

            weighted_sum = 0.0
            weight_sum = 0.0

            for resource, load in result.items():
                if resource in weights:
                    weighted_sum += load * weights[resource]
                    weight_sum += weights[resource]

            if weight_sum > 0:
                result["overall"] = weighted_sum / weight_sum

        return result

    def _generate_recommendations(self, metrics_data: Dict[str, Dict[str, float]],
                                bottlenecks: List[Dict[str, Any]],
                                resource_usage: Dict[str, float],
                                alerts: List[MonitorAlert]) -> List[Dict[str, Any]]:
        """
        Generate system recommendations based on current state.

        Args:
            metrics_data: Dictionary of metrics data
            bottlenecks: List of detected bottlenecks
            resource_usage: Dictionary of resource usage
            alerts: List of active alerts

        Returns:
            List of recommendations
        """
        recommendations = []

        # CPU recommendations
        if "cpu" in resource_usage and resource_usage["cpu"] > 90.0:
            recommendations.append({
                "type": "resource_management",
                "target": "system",
                "priority": "high",
                "action": "reduce_load",
                "description": "Reduce computational load by deferring non-critical tasks"
            })

        # Memory recommendations
        if "memory" in resource_usage and resource_usage["memory"] > 85.0:
            recommendations.append({
                "type": "resource_management",
                "target": "system",
                "priority": "high",
                "action": "free_memory",
                "description": "Free memory by clearing caches or offloading data"
            })

        # Message queue recommendations
        if "message_queue" in resource_usage and resource_usage["message_queue"] > 100:
            recommendations.append({
                "type": "processing",
                "target": "nexus_core",
                "priority": "medium",
                "action": "increase_processing_threads",
                "description": "Increase message processing capacity to reduce queue backlog"
            })

        # Working memory recommendations
        working_memory_load = self.get_current_value("cognitive.working_memory_load")
        if working_memory_load is not None and working_memory_load > 90.0:
            recommendations.append({
                "type": "cognitive",
                "target": "working_memory",
                "priority": "high",
                "action": "consolidate_memory",
                "description": "Consolidate working memory items to reduce cognitive load"
            })

        # Error rate recommendations
        error_rate = self.get_current_value("stability.error_rate")
        if error_rate is not None and error_rate > 5.0:
            recommendations.append({
                "type": "stability",
                "target": "system",
                "priority": "critical",
                "action": "investigate_errors",
                "description": "Investigate and address high error rate across modules"
            })

        # Bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "resource" and bottleneck["resource"] == "cpu":
                # Check if we already have a CPU recommendation
                if not any(r["target"] == "system" and r["action"] == "reduce_load" for r in recommendations):
                    recommendations.append({
                        "type": "resource_management",
                        "target": "system",
                        "priority": "high" if bottleneck["severity"] == "high" else "medium",
                        "action": "reduce_load",
                        "description": "Reduce CPU load by limiting parallel tasks"
                    })

            elif bottleneck["type"] == "cognitive" and bottleneck["resource"] == "working_memory":
                # Check if we already have a working memory recommendation
                if not any(r["target"] == "working_memory" for r in recommendations):
                    recommendations.append({
                        "type": "cognitive",
                        "target": "working_memory",
                        "priority": "high" if bottleneck["severity"] == "high" else "medium",
                        "action": "prune_memory",
                        "description": "Prune less relevant items from working memory"
                    })

        # Add module-specific recommendations based on active alerts
        for alert in alerts:
            if "core.message_queue_size" in alert.metric_id and alert.severity == "error":
                if not any(r["action"] == "increase_processing_threads" for r in recommendations):
                    recommendations.append({
                        "type": "processing",
                        "target": "nexus_core",
                        "priority": "high",
                        "action": "increase_processing_threads",
                        "description": "Increase message processing capacity to address queue overflow"
                    })

            elif "stability.error_rate" in alert.metric_id and alert.severity == "critical":
                if not any(r["action"] == "investigate_errors" for r in recommendations):
                    recommendations.append({
                        "type": "stability",
                        "target": "system",
                        "priority": "critical",
                        "action": "investigate_errors",
                        "description": "Critical error rate detected, investigate immediately"
                    })

        return recommendations

# --- réparation 6.5 : alias du nom attendu par les autres modules
MetaMonitor = CognitiveMonitor
