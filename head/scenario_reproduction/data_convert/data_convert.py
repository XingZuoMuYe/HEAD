import subprocess
import os
from pathlib import Path

def get_project_root() -> Path:
    """获取项目的根路径（假设项目根目录是当前文件的上两级目录）。"""
    curr_path = Path(__file__).resolve()  # 当前文件的绝对路径
    return curr_path.parent  # 项目根路径

if __name__ == "__main__":

    # 获取项目根路径
    base_path = get_project_root()

    scripts = ["osm_scenario.py", "obj_info.py", "dataset_summary.py"]

    # 依次运行每个脚本
    for script in scripts:
        try:
            print(f"Running {script}...")
            subprocess.run(["python", base_path / script], check=True)
            print(f"{script} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}: {e}")
            break
    else:
        print("All scripts executed successfully.")