"""
PolyID - Advanced Performance Monitoring System
Real-time performance tracking and optimization analytics
"""

import time
import psutil
import threading
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None

class PerformanceMonitor:
    """Advanced performance monitoring and analytics system"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Metric storage
        self._metrics = {}
        self._metric_history = deque(maxlen=history_size)
        
        # System monitoring
        self._system_stats = {}
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Performance thresholds
        self._thresholds = {
            'cpu_usage': 80.0,        # %
            'memory_usage': 85.0,     # %
            'prediction_time': 5.0,   # seconds
            'error_rate': 0.1,        # 10%
            'cache_hit_rate': 0.6     # 60%
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start monitoring
        self.start_monitoring()
    
    def record_metric(self, name: str, value: float, unit: str = '', 
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        
        timestamp = time.time()
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        with self._lock:
            # Store current value
            self._metrics[name] = metric
            
            # Add to history
            self._metric_history.append(metric)
    
    def record_prediction_performance(self, prediction_time: float, 
                                    cache_hit: bool, error_occurred: bool,
                                    smiles: str = None):
        """Record prediction-specific performance metrics"""
        
        tags = {'cache_hit': str(cache_hit), 'has_error': str(error_occurred)}
        if smiles:
            tags['smiles_length'] = str(len(smiles))
        
        self.record_metric('prediction_time', prediction_time, 'seconds', tags)
        self.record_metric('cache_hit', 1.0 if cache_hit else 0.0, 'boolean', tags)
        self.record_metric('prediction_error', 1.0 if error_occurred else 0.0, 'boolean', tags)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics"""
        
        with self._lock:
            current_time = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Application metrics from recent history
            recent_metrics = self._get_recent_metrics(60)  # Last 60 seconds
            
            metrics = {
                'timestamp': current_time,
                'system': {
                    'cpu_usage_percent': cpu_percent,
                    'memory_usage_percent': memory.percent,
                    'memory_available_mb': memory.available / 1024 / 1024,
                    'memory_total_mb': memory.total / 1024 / 1024
                },
                'application': {
                    'prediction_count_last_minute': len([m for m in recent_metrics if m.metric_name == 'prediction_time']),
                    'average_prediction_time': self._calculate_average('prediction_time', recent_metrics),
                    'cache_hit_rate': self._calculate_cache_hit_rate(recent_metrics),
                    'error_rate': self._calculate_error_rate(recent_metrics),
                    'throughput_per_second': self._calculate_throughput(recent_metrics)
                },
                'alerts': self._check_alerts()
            }
            
            return metrics
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        with self._lock:
            time_window_seconds = time_window_minutes * 60
            recent_metrics = self._get_recent_metrics(time_window_seconds)
            
            if not recent_metrics:
                return {'error': 'No metrics available for the specified time window'}
            
            # Calculate statistics
            prediction_times = [m.value for m in recent_metrics if m.metric_name == 'prediction_time']
            cache_hits = [m.value for m in recent_metrics if m.metric_name == 'cache_hit']
            errors = [m.value for m in recent_metrics if m.metric_name == 'prediction_error']
            
            summary = {
                'time_window_minutes': time_window_minutes,
                'total_predictions': len(prediction_times),
                'performance': {
                    'average_prediction_time': np.mean(prediction_times) if prediction_times else 0,
                    'median_prediction_time': np.median(prediction_times) if prediction_times else 0,
                    'min_prediction_time': np.min(prediction_times) if prediction_times else 0,
                    'max_prediction_time': np.max(prediction_times) if prediction_times else 0,
                    'prediction_time_std': np.std(prediction_times) if prediction_times else 0
                },
                'reliability': {
                    'cache_hit_rate': np.mean(cache_hits) if cache_hits else 0,
                    'error_rate': np.mean(errors) if errors else 0,
                    'success_rate': 1.0 - (np.mean(errors) if errors else 0)
                },
                'throughput': {
                    'predictions_per_minute': len(prediction_times) / max(time_window_minutes, 1),
                    'predictions_per_second': len(prediction_times) / max(time_window_seconds, 1)
                }
            }
            
            return summary
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on performance data"""
        
        recommendations = []
        current_metrics = self.get_current_metrics()
        
        # CPU usage recommendations
        cpu_usage = current_metrics['system']['cpu_usage_percent']
        if cpu_usage > self._thresholds['cpu_usage']:
            recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'issue': f'High CPU usage: {cpu_usage:.1f}%',
                'recommendation': 'Consider reducing batch size or increasing worker threads'
            })
        
        # Memory usage recommendations
        memory_usage = current_metrics['system']['memory_usage_percent']
        if memory_usage > self._thresholds['memory_usage']:
            recommendations.append({
                'category': 'Memory',
                'priority': 'High',
                'issue': f'High memory usage: {memory_usage:.1f}%',
                'recommendation': 'Clear caches or reduce model memory footprint'
            })
        
        # Prediction time recommendations
        avg_pred_time = current_metrics['application']['average_prediction_time']
        if avg_pred_time > self._thresholds['prediction_time']:
            recommendations.append({
                'category': 'Latency',
                'priority': 'Medium',
                'issue': f'Slow predictions: {avg_pred_time:.2f}s average',
                'recommendation': 'Enable caching or optimize model inference'
            })
        
        # Cache hit rate recommendations
        cache_hit_rate = current_metrics['application']['cache_hit_rate']
        if cache_hit_rate < self._thresholds['cache_hit_rate']:
            recommendations.append({
                'category': 'Caching',
                'priority': 'Medium',
                'issue': f'Low cache hit rate: {cache_hit_rate:.1%}',
                'recommendation': 'Increase cache size or improve cache key strategy'
            })
        
        # Error rate recommendations
        error_rate = current_metrics['application']['error_rate']
        if error_rate > self._thresholds['error_rate']:
            recommendations.append({
                'category': 'Reliability',
                'priority': 'High',
                'issue': f'High error rate: {error_rate:.1%}',
                'recommendation': 'Check input validation and model stability'
            })
        
        return recommendations
    
    def export_metrics(self, format_type: str = 'json', time_window_minutes: int = 60) -> str:
        """Export metrics data in specified format"""
        
        summary = self.get_performance_summary(time_window_minutes)
        current_metrics = self.get_current_metrics()
        recommendations = self.get_optimization_recommendations()
        
        export_data = {
            'export_timestamp': time.time(),
            'time_window_minutes': time_window_minutes,
            'current_metrics': current_metrics,
            'performance_summary': summary,
            'optimization_recommendations': recommendations
        }
        
        if format_type.lower() == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def start_monitoring(self):
        """Start background system monitoring"""
        
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print("[MONITOR] Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        print("[MONITOR] Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        
        while self._monitoring_active:
            try:
                # Record system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.record_metric('system_cpu_usage', cpu_percent, 'percent')
                self.record_metric('system_memory_usage', memory.percent, 'percent')
                self.record_metric('system_memory_available', memory.available / 1024 / 1024, 'mb')
                
                # Sleep between measurements
                time.sleep(10)  # Record every 10 seconds
                
            except Exception as e:
                print(f"[MONITOR] Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _get_recent_metrics(self, seconds: float) -> List[PerformanceMetric]:
        """Get metrics from the last N seconds"""
        
        cutoff_time = time.time() - seconds
        return [m for m in self._metric_history if m.timestamp >= cutoff_time]
    
    def _calculate_average(self, metric_name: str, metrics: List[PerformanceMetric]) -> float:
        """Calculate average value for a specific metric"""
        
        values = [m.value for m in metrics if m.metric_name == metric_name]
        return np.mean(values) if values else 0.0
    
    def _calculate_cache_hit_rate(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate cache hit rate from metrics"""
        
        cache_hits = [m.value for m in metrics if m.metric_name == 'cache_hit']
        return np.mean(cache_hits) if cache_hits else 0.0
    
    def _calculate_error_rate(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate error rate from metrics"""
        
        errors = [m.value for m in metrics if m.metric_name == 'prediction_error']
        return np.mean(errors) if errors else 0.0
    
    def _calculate_throughput(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate throughput (predictions per second)"""
        
        prediction_metrics = [m for m in metrics if m.metric_name == 'prediction_time']
        if not prediction_metrics:
            return 0.0
        
        time_span = max(m.timestamp for m in prediction_metrics) - min(m.timestamp for m in prediction_metrics)
        return len(prediction_metrics) / max(time_span, 1.0)
    
    def _check_alerts(self) -> List[Dict[str, str]]:
        """Check for performance alerts"""
        
        alerts = []
        recent_metrics = self._get_recent_metrics(300)  # Last 5 minutes
        
        # Check prediction time alert
        avg_pred_time = self._calculate_average('prediction_time', recent_metrics)
        if avg_pred_time > self._thresholds['prediction_time']:
            alerts.append({
                'type': 'warning',
                'message': f'Average prediction time ({avg_pred_time:.2f}s) exceeds threshold ({self._thresholds["prediction_time"]}s)'
            })
        
        # Check error rate alert
        error_rate = self._calculate_error_rate(recent_metrics)
        if error_rate > self._thresholds['error_rate']:
            alerts.append({
                'type': 'error',
                'message': f'Error rate ({error_rate:.1%}) exceeds threshold ({self._thresholds["error_rate"]:.1%})'
            })
        
        return alerts
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop_monitoring()
        except:
            pass


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor