"""
GUPPIE Logging and Error Handling System
Enterprise-grade logging and comprehensive error handling
"""

import logging
import sys
import time
import traceback
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
import asyncio
from contextlib import asynccontextmanager


class StructuredLogger:
    """
    Enhanced structured logging system
    """
    
    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.metrics: Dict[str, Any] = {
            "errors": 0,
            "warnings": 0,
            "operations": 0,
            "start_time": time.time()
        }
    
    def log_operation(self, operation: str, **kwargs):
        """Log an operation with structured data"""
        self.metrics["operations"] += 1
        log_data = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.copy(),
            **kwargs
        }
        self.logger.info(f"OPERATION: {operation} | {json.dumps(log_data, default=str)}")
    
    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log an error with full context"""
        self.metrics["errors"] += 1
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.copy(),
            **kwargs
        }
        self.logger.error(f"ERROR: {context} | {json.dumps(error_data, default=str)}")
    
    def log_warning(self, message: str, **kwargs):
        """Log a warning with context"""
        self.metrics["warnings"] += 1
        warning_data = {
            "warning": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.copy(),
            **kwargs
        }
        self.logger.warning(f"WARNING: {message} | {json.dumps(warning_data, default=str)}")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        perf_data = {
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.info(f"PERFORMANCE: {operation} took {duration:.3f}s | {json.dumps(perf_data, default=str)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = time.time() - self.metrics["start_time"]
        return {
            **self.metrics,
            "uptime_seconds": round(uptime, 2),
            "operations_per_second": round(self.metrics["operations"] / uptime, 2) if uptime > 0 else 0
        }


class ErrorHandlingMixin:
    """
    Mixin class for comprehensive error handling
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
        self.error_counts: Dict[str, int] = {}
        self.last_errors: List[Dict[str, Any]] = []
        self.max_error_history = 100
    
    def handle_error(self, error: Exception, context: str = "", reraise: bool = True) -> Optional[Exception]:
        """
        Handle an error with logging and tracking
        
        Args:
            error: The exception that occurred
            context: Context description
            reraise: Whether to reraise the exception
            
        Returns:
            The exception if not reraised
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_record = {
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "count": self.error_counts[error_type],
            "class": self.__class__.__name__
        }
        
        # Add to error history
        self.last_errors.append(error_record)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
        
        # Log the error
        self.logger.log_error(error, context, error_type=error_type, count=self.error_counts[error_type], class_name=self.__class__.__name__)
        
        if reraise:
            raise error
        return error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for this instance"""
        return {
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.last_errors[-10:],  # Last 10 errors
            "total_errors": sum(self.error_counts.values())
        }


def error_handler(context: str = "", reraise: bool = True, log_performance: bool = False):
    """
    Decorator for comprehensive error handling
    
    Args:
        context: Context description
        reraise: Whether to reraise exceptions
        log_performance: Whether to log performance metrics
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            operation_context = context or f"{func.__module__}.{func.__name__}"
            
            try:
                logger.log_operation(f"START: {operation_context}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
                result = await func(*args, **kwargs)
                
                if log_performance:
                    duration = time.time() - start_time
                    logger.log_performance(operation_context, duration)
                
                logger.log_operation(f"SUCCESS: {operation_context}")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.log_error(e, operation_context, duration_ms=round(duration * 1000, 2))
                
                if reraise:
                    raise
                return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            operation_context = context or f"{func.__module__}.{func.__name__}"
            
            try:
                logger.log_operation(f"START: {operation_context}", args_count=len(args), kwargs_keys=list(kwargs.keys()))
                result = func(*args, **kwargs)
                
                if log_performance:
                    duration = time.time() - start_time
                    logger.log_performance(operation_context, duration)
                
                logger.log_operation(f"SUCCESS: {operation_context}")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.log_error(e, operation_context, duration_ms=round(duration * 1000, 2))
                
                if reraise:
                    raise
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@asynccontextmanager
async def error_context(context: str, logger: Optional[StructuredLogger] = None):
    """
    Async context manager for error handling
    
    Args:
        context: Operation context
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger("error_context")
    
    start_time = time.time()
    logger.log_operation(f"START_CONTEXT: {context}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.log_performance(f"CONTEXT: {context}", duration)
        logger.log_operation(f"SUCCESS_CONTEXT: {context}")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.log_error(e, f"CONTEXT: {context}", duration_ms=round(duration * 1000, 2))
        raise


class CircuitBreaker:
    """
    Circuit breaker pattern for handling failures
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: Exception = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to handle
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = get_logger("CircuitBreaker")
    
    def __call__(self, func):
        """Decorator to apply circuit breaker"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                raise ValueError("Cannot use sync wrapper with async function")
            return self._execute_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def _execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.log_operation("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - operation blocked")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _execute_sync(self, func, *args, **kwargs):
        """Execute sync function with circuit breaker logic"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.log_operation("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - operation blocked")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.log_operation("Circuit breaker reset to CLOSED state")
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.log_warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryHandler:
    """
    Exponential backoff retry handler
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, exponential_factor: float = 2.0):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.logger = get_logger("RetryHandler")
    
    def __call__(self, func):
        """Decorator to apply retry logic"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_retry(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                raise ValueError("Cannot use sync wrapper with async function")
            return self._execute_with_retry_sync(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute async function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.logger.log_operation(f"Retry succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.exponential_factor ** attempt), self.max_delay)
                    self.logger.log_warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s", error=str(e))
                    await asyncio.sleep(delay)
                else:
                    self.logger.log_error(e, f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def _execute_with_retry_sync(self, func, *args, **kwargs):
        """Execute sync function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.log_operation(f"Retry succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.exponential_factor ** attempt), self.max_delay)
                    self.logger.log_warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s", error=str(e))
                    time.sleep(delay)
                else:
                    self.logger.log_error(e, f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> StructuredLogger:
    """
    Get or create a structured logger
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        if log_file is None:
            log_file = f"/tmp/guppie_logs/{name}.log"
        _loggers[name] = StructuredLogger(name, log_level, log_file)
    
    return _loggers[name]


def get_all_metrics() -> Dict[str, Any]:
    """Get metrics from all loggers"""
    return {name: logger.get_metrics() for name, logger in _loggers.items()}


# Convenience functions
def log_operation(operation: str, logger_name: str = "guppie", **kwargs):
    """Log an operation"""
    logger = get_logger(logger_name)
    logger.log_operation(operation, **kwargs)


def log_error(error: Exception, context: str = "", logger_name: str = "guppie", **kwargs):
    """Log an error"""
    logger = get_logger(logger_name)
    logger.log_error(error, context, **kwargs)


def log_performance(operation: str, duration: float, logger_name: str = "guppie", **kwargs):
    """Log performance metrics"""
    logger = get_logger(logger_name)
    logger.log_performance(operation, duration, **kwargs)