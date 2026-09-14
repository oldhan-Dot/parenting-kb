from pathlib import Path
import os

from dotenv import load_dotenv


def get_path_dir(ps: int = 0) -> Path:
    """
    通过 pathlib 的 parents 快捷取上级目录
    parents[0] = 上一级，parents[1] = 上两级，以此类推
    """
    return Path(__file__).parents[ps]


def get_project_root(identifier: str = ".env") -> Path:
    """
    向上查找项目根目录（以 .env 作为标识文件）
    优先读环境变量 PROJECT_ROOT，其次查找 .env 并顺带加载
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量 PROJECT_ROOT 未配置")


PROJECT_ROOT = get_project_root(".env")
