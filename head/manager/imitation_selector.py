from pathlib import Path
from omegaconf import OmegaConf
from head.policy.imitation_policy.models import build_model
from head.policy.imitation_policy.utils.utils import set_seed
from omegaconf import OmegaConf

def resolve_imitation_strategy(cfg):
    """
    从外部 cfg 中读取 imitation method 名字，并加载对应的内部 config。
    """
    # 1. 从外层配置拿 method 名字
    method_name = cfg.args.algorithm.imitation_learning.algorithm_type.main
    print(f"[外层配置指定的 imitation method] {method_name}")

    # 2. 找到对应的内部 config 文件
    _THIS_FILE = Path(__file__).resolve()
    METHOD_CONFIG_DIR = _THIS_FILE.parent.parent / "policy" / "imitation_policy" / "configs" / "method"
    GLOBAL_CONFIG_DIR = _THIS_FILE.parent.parent / "policy" / "imitation_policy" / "configs"
    method_cfg_path = METHOD_CONFIG_DIR / f"{method_name}.yaml"
    global_cfg_path = GLOBAL_CONFIG_DIR / "config.yaml"
    if not method_cfg_path.exists():
        raise FileNotFoundError(f"未找到 method 配置文件: {method_cfg_path}")

    # 3. 加载内部配置
    method_cfg = OmegaConf.load(method_cfg_path)
    global_cfg = OmegaConf.load(global_cfg_path)
    merged_cfg = OmegaConf.merge({"method": method_cfg }, method_cfg, global_cfg)
    merged_cfg["eval"] = True
    set_seed(merged_cfg.seed)

    # 5. 构建模型
    model = build_model(merged_cfg)
    print(f"[✅ 使用的 method config] {method_cfg_path}")
    return model, merged_cfg
