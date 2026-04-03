import os
import sys


def ensure_local_dependency_paths():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra_paths = [
        os.path.join(repo_dir, "src", "taming-transformers"),
        os.path.join(repo_dir, "src", "clip"),
    ]
    for path in extra_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
