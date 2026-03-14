import os
from pathlib import Path
from dotenv import load_dotenv

_LOADED = False

def load_env():
    """프로젝트 루트의 .env를 1회만 로드"""
    global _LOADED
    if _LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    _LOADED = True


def get_env(key: str, required: bool = True):
    load_env()
    value = os.getenv(key)
    if required and not value:
        raise ValueError(f"Missing environment variable: {key}")
    return value
