"""
PolyID - Intelligent Model Caching and Warm-up System
High-performance caching for model predictions and preprocessing
"""

import hashlib
import pickle
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

class ModelCacheManager:
    """Intelligent caching system for model predictions and preprocessing"""

    def __init__(self, cache_dir: str = "model_cache", max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_items = max_memory_items

        # In-memory cache for fast access
        self._memory_cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._warm_up_complete = False

        # Clean up old cache files on startup
        self._cleanup_old_cache()

    def get_cache_key(self, smiles: str, properties: List[str], **kwargs) -> str:
        """Generate unique cache key for prediction"""
        # Include all relevant parameters in cache key
        content_parts = [
            smiles.strip(),
            str(sorted(properties)),
            str(kwargs.get('model_version', 'v1')),
            str(kwargs.get('prediction_params', {}))
        ]
        content = "|".join(content_parts)
        return hashlib.md5(content.encode()).hexdigest()

    def cache_prediction(self, key: str, result: Dict, metadata: Optional[Dict] = None) -> None:
        """Cache prediction result with metadata"""
        with self._lock:
            cache_entry = {
                'result': result,
                'timestamp': time.time(),
                'metadata': metadata or {},
                'access_count': 0
            }

            # Store in memory cache
            self._memory_cache[key] = cache_entry
            self._cache_timestamps[key] = cache_entry['timestamp']

            # Limit memory cache size
            if len(self._memory_cache) > self.max_memory_items:
                self._evict_oldest()

            # Optionally persist to disk for long-term caching
            if metadata and metadata.get('persist', False):
                self._persist_to_disk(key, cache_entry)

    def get_cached_prediction(self, key: str, max_age_seconds: int = 3600) -> Optional[Dict]:
        """Retrieve cached prediction if valid"""
        with self._lock:
            if key in self._memory_cache:
                cache_entry = self._memory_cache[key]
                cache_age = time.time() - cache_entry['timestamp']

                # Check if cache is still valid
                if cache_age < max_age_seconds:
                    cache_entry['access_count'] += 1
                    self._hits += 1
                    return cache_entry['result']
                else:
                    # Remove expired cache entry
                    del self._memory_cache[key]
                    del self._cache_timestamps[key]

            self._misses += 1
            return None

    def warm_up_models(self, sample_data: Optional[List[Dict]] = None) -> bool:
        """Pre-load and warm up models with sample data"""
        if self._warm_up_complete:
            return True

        try:
            print("[CACHE] Warming up PolyID models...")

            # Default sample polymers for warm-up
            if sample_data is None:
                sample_data = [
                    {'smiles': 'CC', 'properties': ['Tg']},  # Polyethylene
                    {'smiles': 'CC(C)', 'properties': ['Tg', 'Tm']},  # Polypropylene
                    {'smiles': 'CC(c1ccccc1)', 'properties': ['Tg']},  # Polystyrene
                ]

            # Import here to avoid circular imports
            from polyid.optimized_predictor import OptimizedPredictor
            predictor = OptimizedPredictor()

            # Warm up with sample predictions
            for sample in sample_data:
                try:
                    result = predictor.predict_properties(
                        sample['smiles'],
                        sample['properties']
                    )
                    if 'error' not in result:
                        cache_key = self.get_cache_key(**sample)
                        self.cache_prediction(cache_key, result, {'warmup': True})
                        print(f"[CACHE] Warmed up: {sample['smiles']}")
                except Exception as e:
                    print(f"[CACHE] Warm-up failed for {sample['smiles']}: {e}")

            self._warm_up_complete = True
            print("[CACHE] Model warm-up completed successfully")
            return True

        except Exception as e:
            print(f"[CACHE] Model warm-up failed: {e}")
            return False

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0

        return {
            'memory_cache_size': len(self._memory_cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'warm_up_complete': self._warm_up_complete
        }

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally matching a pattern"""
        with self._lock:
            if pattern:
                # Clear specific pattern (not implemented for simplicity)
                cleared = 0
            else:
                cleared = len(self._memory_cache)
                self._memory_cache.clear()
                self._cache_timestamps.clear()
                self._hits = 0
                self._misses = 0

            # Clear disk cache
            self._cleanup_old_cache()
            return cleared

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries when memory limit is reached"""
        if not self._cache_timestamps:
            return

        # Find oldest entry
        oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)

        # Remove from both caches
        del self._memory_cache[oldest_key]
        del self._cache_timestamps[oldest_key]

    def _persist_to_disk(self, key: str, cache_entry: Dict) -> None:
        """Persist cache entry to disk for long-term storage"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
        except Exception:
            # Silently fail if disk caching fails
            pass

    def _load_from_disk(self, key: str) -> Optional[Dict]:
        """Load cache entry from disk"""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None

    def _cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """Clean up old cache files from disk"""
        try:
            max_age_seconds = max_age_days * 24 * 60 * 60
            current_time = time.time()

            for cache_file in self.cache_dir.glob("*.pkl"):
                if current_time - cache_file.stat().st_mtime > max_age_seconds:
                    cache_file.unlink()
        except Exception:
            # Silently fail if cleanup fails
            pass

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> ModelCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager()
    return _cache_manager