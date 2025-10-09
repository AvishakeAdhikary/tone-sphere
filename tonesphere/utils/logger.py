"""
Modern Logging System for ToneSphere
Supports console and file logging with rotation, structured logging, and performance tracking
"""
import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for machine-readable logs
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for better readability
    """
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Build message
        message = f"{color}[{record.levelname}]{reset} {timestamp} - {record.name} - {record.getMessage()}"
        
        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class ToneSphereLogger:
    """
    Modern logging manager for ToneSphere
    Supports console and file logging with rotation
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.loggers: Dict[str, logging.Logger] = {}
        self.file_logging_enabled = False
        self.log_file_path: Optional[Path] = None
        self.structured_logging = False
    
    def setup_logger(
        self,
        name: str = "tonesphere",
        level: int = logging.INFO,
        enable_file_logging: bool = False,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        structured: bool = False,
        console_colors: bool = True
    ) -> logging.Logger:
        """
        Setup and configure logger with modern features
        
        Args:
            name: Logger name
            level: Logging level
            enable_file_logging: Enable file logging
            log_file: Log file name (default: tonesphere.log)
            log_dir: Log directory (default: ./logs)
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep
            structured: Use structured JSON logging
            console_colors: Use colored console output
            
        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if console_colors and not structured:
            console_formatter = ColoredConsoleFormatter()
        elif structured:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file_logging:
            self.file_logging_enabled = True
            self.structured_logging = structured
            
            # Setup log directory
            if log_dir is None:
                log_dir = Path.cwd() / "logs"
            else:
                log_dir = Path(log_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup log file
            if log_file is None:
                log_file = "tonesphere.log"
            
            self.log_file_path = log_dir / log_file
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            # Use structured format for file logs if requested
            if structured:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"File logging enabled: {self.log_file_path}")
        
        self.loggers[name] = logger
        return logger
    
    def get_logger(self, name: str = "tonesphere") -> logging.Logger:
        """Get or create logger"""
        if name not in self.loggers:
            return self.setup_logger(name)
        return self.loggers[name]
    
    def enable_file_logging(
        self,
        log_file: str = "tonesphere.log",
        log_dir: Optional[str] = None,
        level: int = logging.INFO
    ):
        """Enable file logging for all existing loggers"""
        for logger_name, logger in self.loggers.items():
            # Remove old file handlers
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    logger.removeHandler(handler)
            
            # Add new file handler
            if log_dir is None:
                log_dir = Path.cwd() / "logs"
            else:
                log_dir = Path(log_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = log_dir / log_file
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.file_logging_enabled = True
        logger.info(f"File logging enabled: {self.log_file_path}")
    
    def disable_file_logging(self):
        """Disable file logging for all loggers"""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    logger.removeHandler(handler)
        
        self.file_logging_enabled = False
    
    def set_level(self, level: int, logger_name: Optional[str] = None):
        """Set logging level"""
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(level)
        else:
            for logger in self.loggers.values():
                logger.setLevel(level)
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get current log file path"""
        return self.log_file_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'file_logging_enabled': self.file_logging_enabled,
            'log_file_path': str(self.log_file_path) if self.log_file_path else None,
            'structured_logging': self.structured_logging,
            'active_loggers': len(self.loggers),
            'logger_names': list(self.loggers.keys())
        }
        
        if self.log_file_path and self.log_file_path.exists():
            stats['log_file_size'] = self.log_file_path.stat().st_size
        
        return stats


# Global logger manager instance
logger_manager = ToneSphereLogger()

# Default logger for backward compatibility
logger = logger_manager.setup_logger()


def get_logger(name: str = "tonesphere") -> logging.Logger:
    """Get or create a logger"""
    return logger_manager.get_logger(name)


def enable_file_logging(log_file: str = "tonesphere.log", log_dir: Optional[str] = None):
    """Enable file logging"""
    logger_manager.enable_file_logging(log_file, log_dir)


def disable_file_logging():
    """Disable file logging"""
    logger_manager.disable_file_logging()


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics"""
    return logger_manager.get_stats()