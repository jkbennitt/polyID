"""
PolyID - Asynchronous Batch Processing System
High-performance concurrent processing for polymer property predictions
"""

import asyncio
import concurrent.futures
import threading
import queue
import time
import json
from typing import List, Dict, Any, Callable, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

class AsyncPredictionProcessor:
    """High-performance asynchronous processing system for polymer predictions"""

    def __init__(self, max_workers: int = 4, max_batch_size: int = 50):
        self.max_workers = max_workers
        self.max_batch_size = max_batch_size
        
        # Thread pool for CPU-intensive tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Task management
        self._task_queue = queue.PriorityQueue()
        self._results_cache = {}
        self._active_tasks = {}
        self._task_counter = 0
        
        # Performance monitoring
        self._processed_count = 0
        self._total_processing_time = 0.0
        self._error_count = 0
        
        # Thread safety
        self._lock = threading.RLock()

    async def predict_async(self, smiles: str, properties: List[str], 
                          priority: int = 1, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process single prediction asynchronously"""
        
        task_id = f"single_{self._task_counter}"
        self._task_counter += 1
        
        # Submit to thread pool
        loop = asyncio.get_event_loop()
        future = self.executor.submit(self._predict_single_sync, smiles, properties, task_id)
        
        try:
            # Wait for completion asynchronously
            result = await loop.run_in_executor(None, future.result)
            
            if callback:
                callback(task_id, result)
                
            return result
            
        except Exception as e:
            error_result = {'error': f'Async prediction failed: {str(e)}', 'task_id': task_id}
            if callback:
                callback(task_id, error_result)
            return error_result

    async def predict_batch_async(self, batch_data: List[Dict[str, Any]], 
                                progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Process multiple predictions concurrently with progress tracking"""
        
        if not batch_data:
            return []
            
        batch_id = f"batch_{int(time.time())}"
        total_items = len(batch_data)
        
        # Split large batches into smaller chunks for better performance
        chunks = self._split_into_chunks(batch_data, self.max_batch_size)
        
        results = []
        processed = 0
        
        # Process chunks concurrently
        chunk_futures = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{batch_id}_chunk_{i}"
            future = self.executor.submit(self._process_chunk_sync, chunk, chunk_id)
            chunk_futures.append(future)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(chunk_futures):
            try:
                chunk_results = future.result(timeout=300)  # 5-minute timeout per chunk
                results.extend(chunk_results)
                processed += len(chunk_results)
                
                # Update progress
                if progress_callback:
                    progress = processed / total_items
                    progress_callback(progress, processed, total_items)
                    
            except concurrent.futures.TimeoutError:
                self._error_count += len(chunks[0])  # Estimate chunk size
                chunk_error = {'error': 'Chunk processing timeout', 'batch_id': batch_id}
                results.extend([chunk_error] * len(chunks[0]))
            except Exception as e:
                self._error_count += 1
                error_result = {'error': f'Chunk processing failed: {str(e)}', 'batch_id': batch_id}
                results.append(error_result)
        
        return results[:total_items]  # Ensure we return exactly the expected number

    def predict_batch_sync(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous batch processing for compatibility"""
        
        if not batch_data:
            return []
            
        # Process in parallel using thread pool
        chunk_size = min(self.max_batch_size, max(1, len(batch_data) // self.max_workers))
        chunks = self._split_into_chunks(batch_data, chunk_size)
        
        futures = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"sync_batch_{i}_{int(time.time())}"
            future = self.executor.submit(self._process_chunk_sync, chunk, chunk_id)
            futures.append(future)
        
        # Collect all results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_results = future.result(timeout=180)  # 3-minute timeout
                results.extend(chunk_results)
            except Exception as e:
                error_result = {'error': f'Sync batch processing failed: {str(e)}'}
                results.append(error_result)
        
        return results

    def process_csv_file(self, file_content: Union[str, bytes], 
                        smiles_column: str = 'smiles_polymer',
                        properties: List[str] = None,
                        progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Process batch predictions from CSV content"""
        
        try:
            # Parse CSV content
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
                
            import io
            df = pd.read_csv(io.StringIO(file_content))
            
            if smiles_column not in df.columns:
                raise ValueError(f"Column '{smiles_column}' not found in CSV")
            
            # Default properties if not specified
            if properties is None:
                properties = ['Glass Transition Temperature (Tg)', 'Melting Temperature (Tm)']
            
            # Prepare batch data
            batch_data = []
            for idx, row in df.iterrows():
                smiles = row[smiles_column]
                if pd.notna(smiles) and smiles.strip():
                    batch_data.append({
                        'smiles': smiles.strip(),
                        'properties': properties,
                        'original_index': idx
                    })
            
            if not batch_data:
                raise ValueError("No valid SMILES found in the data")
            
            # Process batch
            results = self.predict_batch_sync(batch_data)
            
            # Convert results back to DataFrame format
            results_df = self._format_batch_results(df, results, properties)
            
            return results_df
            
        except Exception as e:
            raise ValueError(f"CSV batch processing failed: {str(e)}")

    def export_results(self, df: pd.DataFrame, format_type: str = 'csv') -> bytes:
        """Export results in specified format"""
        
        if format_type.lower() == 'csv':
            return df.to_csv(index=False).encode('utf-8')
            
        elif format_type.lower() == 'xlsx':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='PolyID_Results')
                
                # Add metadata sheet
                metadata_df = pd.DataFrame([
                    ['Processing Date', time.strftime('%Y-%m-%d %H:%M:%S')],
                    ['Total Predictions', len(df)],
                    ['PolyID Version', 'Optimized v2.0'],
                    ['Processing Mode', 'Async Batch']
                ], columns=['Parameter', 'Value'])
                metadata_df.to_excel(writer, index=False, sheet_name='Metadata')
                
            return output.getvalue()
            
        elif format_type.lower() == 'json':
            return df.to_json(orient='records', indent=2).encode('utf-8')
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        with self._lock:
            avg_time = (self._total_processing_time / self._processed_count 
                       if self._processed_count > 0 else 0)
            
            return {
                'processed_count': self._processed_count,
                'error_count': self._error_count,
                'success_rate': ((self._processed_count - self._error_count) / 
                               max(self._processed_count, 1)),
                'average_processing_time': avg_time,
                'total_processing_time': self._total_processing_time,
                'active_tasks': len(self._active_tasks),
                'max_workers': self.max_workers,
                'max_batch_size': self.max_batch_size,
                'throughput_per_second': (self._processed_count / 
                                        max(self._total_processing_time, 1))
            }

    def _predict_single_sync(self, smiles: str, properties: List[str], task_id: str) -> Dict[str, Any]:
        """Synchronous single prediction wrapper"""
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from polyid.optimized_predictor import OptimizedPredictor
            
            predictor = OptimizedPredictor()
            result = predictor.predict_properties(smiles, properties)
            
            # Add task metadata
            result['task_id'] = task_id
            result['processing_time'] = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self._processed_count += 1
                self._total_processing_time += result['processing_time']
            
            return result
            
        except Exception as e:
            error_result = {
                'error': f'Single prediction failed: {str(e)}',
                'task_id': task_id,
                'processing_time': time.time() - start_time
            }
            
            with self._lock:
                self._error_count += 1
                
            return error_result

    def _process_chunk_sync(self, chunk: List[Dict[str, Any]], chunk_id: str) -> List[Dict[str, Any]]:
        """Process a chunk of predictions synchronously"""
        
        results = []
        chunk_start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from polyid.optimized_predictor import OptimizedPredictor
            
            predictor = OptimizedPredictor()
            
            for item in chunk:
                try:
                    smiles = item.get('smiles', '')
                    properties = item.get('properties', [])
                    
                    if not smiles or not properties:
                        result = {'error': 'Missing SMILES or properties', 'chunk_id': chunk_id}
                    else:
                        result = predictor.predict_properties(smiles, properties)
                        result['chunk_id'] = chunk_id
                        result['original_index'] = item.get('original_index')
                    
                    results.append(result)
                    
                except Exception as e:
                    error_result = {
                        'error': f'Item processing failed: {str(e)}',
                        'chunk_id': chunk_id,
                        'original_index': item.get('original_index')
                    }
                    results.append(error_result)
            
            # Update chunk statistics
            chunk_time = time.time() - chunk_start_time
            with self._lock:
                self._processed_count += len(results)
                self._total_processing_time += chunk_time
                
        except Exception as e:
            # If entire chunk fails, return error for all items
            error_result = {'error': f'Chunk processing failed: {str(e)}', 'chunk_id': chunk_id}
            results = [error_result.copy() for _ in chunk]
            
            with self._lock:
                self._error_count += len(chunk)
        
        return results

    def _split_into_chunks(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split data into chunks for parallel processing"""
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks

    def _format_batch_results(self, original_df: pd.DataFrame, 
                             results: List[Dict[str, Any]], 
                             properties: List[str]) -> pd.DataFrame:
        """Format batch processing results back to DataFrame"""
        
        # Create results DataFrame
        result_data = []
        
        for result in results:
            row_data = {}
            original_idx = result.get('original_index', 0)
            
            # Copy original data if available
            if original_idx < len(original_df):
                row_data.update(original_df.iloc[original_idx].to_dict())
            
            # Add prediction results
            if 'error' in result:
                for prop in properties:
                    row_data[f"{prop}_pred"] = np.nan
                    row_data[f"{prop}_confidence"] = 'Error'
                row_data['prediction_error'] = result['error']
            else:
                for prop in properties:
                    if prop in result:
                        prop_data = result[prop]
                        if isinstance(prop_data, dict):
                            row_data[f"{prop}_pred"] = prop_data.get('value', np.nan)
                            row_data[f"{prop}_confidence"] = prop_data.get('confidence', 'Unknown')
                        else:
                            row_data[f"{prop}_pred"] = prop_data
                            row_data[f"{prop}_confidence"] = 'Medium'
                    else:
                        row_data[f"{prop}_pred"] = np.nan
                        row_data[f"{prop}_confidence"] = 'Missing'
                
                # Add processing metadata
                row_data['processing_time'] = result.get('processing_time', 0)
                row_data['chunk_id'] = result.get('chunk_id', '')
            
            result_data.append(row_data)
        
        return pd.DataFrame(result_data)

    def shutdown(self):
        """Gracefully shutdown the processor"""
        
        print("[ASYNC] Shutting down async processor...")
        self.executor.shutdown(wait=True)
        
        with self._lock:
            self._active_tasks.clear()
            self._results_cache.clear()
        
        print("[ASYNC] Async processor shutdown complete")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup


# Global async processor instance
_async_processor = None

def get_async_processor() -> AsyncPredictionProcessor:
    """Get global async processor instance"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncPredictionProcessor()
    return _async_processor