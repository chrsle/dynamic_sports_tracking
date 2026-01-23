"""
Gunicorn configuration for production deployment.

Usage:
    gunicorn -c gunicorn.conf.py dashboard.server:app

Environment Variables:
    WORKERS: Number of worker processes (default: CPU cores * 2 + 1)
    PORT: Port to bind to (default: 8000)
    HOST: Host to bind to (default: 0.0.0.0)
    LOG_LEVEL: Logging level (default: info)
"""
import multiprocessing
import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
timeout = 120
keepalive = 5

# Process naming
proc_name = 'hockey-analytics'

# Logging
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '{"timestamp": "%(t)s", "method": "%(m)s", "path": "%(U)s", "status": "%(s)s", "response_time": "%(D)s", "size": "%(B)s", "remote_addr": "%(h)s"}'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Graceful shutdown
graceful_timeout = 30

# Preload app for memory efficiency
preload_app = True


def on_starting(server):
    """Called just before the master process is initialized."""
    pass


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    pass


def when_ready(server):
    """Called just after the server is started."""
    pass


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    pass


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    pass


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    pass


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    pass


def pre_exec(server):
    """Called just before a new master process is forked."""
    pass


def child_exit(server, worker):
    """Called when a worker process is terminated."""
    pass


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    pass


def nworkers_changed(server, new_value, old_value):
    """Called when the number of workers changes."""
    pass


def on_exit(server):
    """Called just before exiting gunicorn."""
    pass
