"""
Logging utilities for VMM-FRC application.

Provides centralized logging to both console and file, with GUI access.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import os


class VMMLLogger:
    """Centralized logger for VMM-FRC application."""

    _instance = None
    _log_file = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VMMLLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize logger with file and console handlers."""
        # Create logs directory
        log_dir = Path.home() / '.vmm-frc' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_file = log_dir / f'vmm_log_{timestamp}.txt'

        # Create logger
        self._logger = logging.getLogger('VMM-FRC')
        self._logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if self._logger.handlers:
            self._logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)

        # Console handler (for backward compatibility with print statements)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        self._logger.info(f"VMM-FRC logging initialized: {self._log_file}")

    def get_logger(self):
        """Get the logger instance."""
        return self._logger

    def get_log_file(self):
        """Get the current log file path."""
        return self._log_file

    def get_recent_logs(self, n_lines=500):
        """Get the most recent n lines from the log file."""
        if not self._log_file or not self._log_file.exists():
            return "No log file available."

        try:
            with open(self._log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return ''.join(lines[-n_lines:])
        except Exception as e:
            return f"Error reading log file: {e}"

    def get_all_logs(self):
        """Get all logs from the current log file."""
        if not self._log_file or not self._log_file.exists():
            return "No log file available."

        try:
            with open(self._log_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {e}"

    @classmethod
    def get_log_directory(cls):
        """Get the logs directory path."""
        return Path.home() / '.vmm-frc' / 'logs'


# Global logger instance
_vmm_logger = VMMLLogger()


def get_logger():
    """Get the VMM-FRC logger instance."""
    return _vmm_logger.get_logger()


def get_log_file():
    """Get the current log file path."""
    return _vmm_logger.get_log_file()


def get_recent_logs(n_lines=500):
    """Get recent log entries."""
    return _vmm_logger.get_recent_logs(n_lines)


def get_all_logs():
    """Get all log entries."""
    return _vmm_logger.get_all_logs()


def get_log_directory():
    """Get the logs directory."""
    return _vmm_logger.get_log_directory()


# Convenience functions for different log levels
def debug(msg):
    """Log debug message."""
    get_logger().debug(msg)


def info(msg):
    """Log info message."""
    get_logger().info(msg)


def warning(msg):
    """Log warning message."""
    get_logger().warning(msg)


def error(msg):
    """Log error message."""
    get_logger().error(msg)


def critical(msg):
    """Log critical message."""
    get_logger().critical(msg)
