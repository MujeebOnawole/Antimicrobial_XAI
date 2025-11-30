# memory_tracker.py
import os
import psutil
import torch
from datetime import datetime
import platform
import sys
import pytorch_lightning as pl
from logger import get_logger
import time
from contextlib import contextmanager

logger = get_logger(__name__)

class MemoryTracker:
    def __init__(self):
        self.start_time = datetime.now().replace(microsecond=0)
        self.start_mem = psutil.Process(os.getpid()).memory_info().rss
        self.peak_mem = self.start_mem
        self.peak_gpu_mem = 0 if torch.cuda.is_available() else None
        self.error_count = 0
        self.max_errors = 3
        self.retry_delay = 5
        self.memory_threshold = 0.85  # 85% memory threshold
        
    def log_system_info(self):
        """Log system information at startup"""
        logger.info(f"[..BEGIN..] {self.start_time: %d-%m-%Y %H:%M:%S}")
        logger.info(f"[..BEGIN..] {self.start_mem/1000000:.1f} MB")
        logger.info(f"[...RUN...] {sys.argv}")
        logger.info(f"[Platform ] {platform.system()}")
        logger.info(f"[Python   ] {sys.version.split('|')[0]}")
        logger.info(f"[Pytorch  ] {torch.__version__}")
        logger.info(f"[Lightning] {pl.__version__}")
        
        if torch.cuda.is_available():    
            logger.info(f"[Cuda     ] {torch.cuda.get_device_name(0)}")
            logger.info(f"[GPU Mem  ] Total: {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f} MB")
        else:
            logger.info(f"[Cuda     ] No Cuda device available")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[Device   ] Using device: {self.device}")

    @contextmanager
    def safe_memory_operation(self, operation_name: str):
        """Context manager for safe memory operations with automatic cleanup."""
        try:
            self.log_memory_stats(f"BEFORE_{operation_name}")
            yield
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM during {operation_name}: {str(e)}")
                self.emergency_cleanup()
                raise
        except Exception as e:
            logger.error(f"Error during {operation_name}: {str(e)}")
            raise
        finally:
            self.log_memory_stats(f"AFTER_{operation_name}")
        
    def log_memory_stats(self, stage: str = ""):
        """Log current memory statistics"""
        current_mem = psutil.Process(os.getpid()).memory_info().rss
        self.peak_mem = max(self.peak_mem, current_mem)
        
        mem_info = {
            "Current CPU": f"{current_mem/1000000:.1f} MB",
            "Peak CPU": f"{self.peak_mem/1000000:.1f} MB",
            "CPU Util": f"{psutil.cpu_percent()}%"
        }
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_allocated()
            self.peak_gpu_mem = max(self.peak_gpu_mem, current_gpu)
            mem_info.update({
                "Current GPU": f"{current_gpu/1024**2:.1f} MB",
                "Peak GPU": f"{self.peak_gpu_mem/1024**2:.1f} MB",
            })
            
            # Try to get GPU utilization if pynvml is available
            try:
                gpu_util = torch.cuda.utilization()
                mem_info["GPU Util"] = f"{gpu_util}%"
            except:
                # Skip GPU utilization if pynvml is not available
                pass
        
        logger.info(f"[{stage}] Memory Stats:")
        for key, value in mem_info.items():
            logger.info(f"  {key}: {value}")

    def check_memory_usage(self):
        """Check current memory usage and return status with details."""
        if not torch.cuda.is_available():
            return True, {}
            
        try:
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            
            usage_ratio = memory_reserved / max_memory
            
            memory_stats = {
                'allocated_gb': memory_allocated / 1024**3,
                'reserved_gb': memory_reserved / 1024**3,
                'total_gb': max_memory / 1024**3,
                'usage_ratio': usage_ratio
            }
            
            is_safe = usage_ratio < self.memory_threshold
            
            return is_safe, memory_stats
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
            return True, {}
            
    def log_classification_memory_stats(self, stage: str = ""):
        """Log detailed memory statistics specifically for classification tasks."""
        if not torch.cuda.is_available():
            return
            
        current_mem = psutil.Process(os.getpid()).memory_info().rss
        self.peak_mem = max(self.peak_mem, current_mem)
        
        gpu_stats = {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_reserved': torch.cuda.max_memory_reserved(),
            'cached': torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        }
        
        logger.info(f"[{stage}] Classification Memory Details:")
        logger.info(f"  CPU Current: {current_mem/1000000:.1f} MB")
        logger.info(f"  CPU Peak: {self.peak_mem/1000000:.1f} MB")
        
        for key, value in gpu_stats.items():
            logger.info(f"  GPU {key}: {value/1024**2:.1f} MB")
            
        # Try to get GPU utilization
        try:
            gpu_util = torch.cuda.utilization()
            logger.info(f"  GPU Utilization: {gpu_util}%")
        except:
            pass

    def safe_cuda_operation(self, operation_func, *args, **kwargs):
        """Execute CUDA operations with safety checks and automatic retries."""
        for attempt in range(self.max_errors):
            try:
                is_safe, memory_stats = self.check_memory_usage()
                
                if not is_safe:
                    logger.warning(
                        f"High memory usage detected: "
                        f"{memory_stats['usage_ratio']:.1%} "
                        f"({memory_stats['reserved_gb']:.1f}GB / {memory_stats['total_gb']:.1f}GB)"
                    )
                    self.emergency_cleanup()
                    
                with self.safe_memory_operation(f"ATTEMPT_{attempt + 1}"):
                    result = operation_func(*args, **kwargs)
                    return result, None
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    self.error_count += 1
                    logger.error(f"CUDA error on attempt {attempt + 1}: {str(e)}")
                    
                    if self.error_count > self.max_errors:
                        logger.error("Maximum CUDA errors exceeded. Aborting operation.")
                        return None, e
                        
                    if attempt < self.max_errors - 1:
                        self.emergency_cleanup()
                        time.sleep(self.retry_delay)
                        continue
                return None, e
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return None, e
                
        return None, RuntimeError(f"Failed after {self.max_errors} attempts")
            
    def cleanup_classification_memory(self, stage: str = ""):
        """Perform memory cleanup for classification tasks and log the results."""
        if not torch.cuda.is_available():
            return
            
        # Log memory before cleanup
        logger.info(f"[{stage}] Memory before cleanup:")
        self.log_classification_memory_stats("BEFORE_CLEANUP")
        
        # Perform cleanup
        torch.cuda.empty_cache()
        
        # Log memory after cleanup
        logger.info(f"[{stage}] Memory after cleanup:")
        self.log_classification_memory_stats("AFTER_CLEANUP")

    def emergency_cleanup(self):
        """Perform emergency memory cleanup with enhanced logging."""
        logger.info("Initiating emergency memory cleanup...")
        
        # Log pre-cleanup state
        self.log_classification_memory_stats("PRE_CLEANUP")
        
        # Perform cleanup operations
        torch.cuda.empty_cache()
        
        if hasattr(torch.cuda, 'memory_summary'):
            logger.debug(f"Memory Summary:\n{torch.cuda.memory_summary()}")
            
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache again after GC
        torch.cuda.empty_cache()
        
        # Log post-cleanup state
        self.log_classification_memory_stats("POST_CLEANUP")
        
    def log_final_stats(self):
        """Log final statistics when program ends"""
        end_time = datetime.now().replace(microsecond=0)
        end_mem = psutil.Process(os.getpid()).memory_info().rss
        
        logger.info(f"[...END...] {end_time: %d-%m-%Y %H:%M:%S}")
        logger.info(f"[...END...] {end_mem/1000000:.1f} MB")
        logger.info(f"[...USE...] Time: {end_time-self.start_time}, Memory: {(end_mem-self.start_mem)/1000000:.1f} MB")
        logger.info(f"[..PEAK...] CPU: {self.peak_mem/1000000:.1f} MB")
        
        if torch.cuda.is_available():
            logger.info(f"[..PEAK...] GPU: {self.peak_gpu_mem/1024**2:.1f} MB")

if __name__ == "__main__":
    memory_tracker = MemoryTracker()
    memory_tracker.log_system_info()