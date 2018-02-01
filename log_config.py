from logging.handlers import RotatingFileHandler
from logging.config import dictConfig


LOG_FILE_SIZE = 50 * 1024 * 1024
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        'standard': {
            'format': '%(asctime)s [%(module)s:%(lineno)s]' \
                      '%(levelname)s: %(message)s'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "./log/recommendation.log",
            "maxBytes": LOG_FILE_SIZE,
            "backupCount": 72,
            "encoding": "utf8"
        },
        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": "./log/recommendation-errors.log",
            "maxBytes": LOG_FILE_SIZE,
            "backupCount": 72,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "my_module": {
            "level": "ERROR",
            "handlers": ["console"],
            "propagate": "no"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "info_file_handler", "error_file_handler"]
    }
}

