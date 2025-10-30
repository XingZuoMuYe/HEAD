#!/bin/bash

#export PYTHONPATH=/home/test/git_shuo/HEAD

# 加载 Conda 初始化代码
source ~/.bashrc
eval "$(conda shell.bash hook)"

# 切换到虚拟环境 metadrive
conda activate HEAD

# 进入指定目录
#cd /home/test/git_shuo/HEAD/scripts

# 提示用户确认配置文件是否已修改
read -p "请确认配置文件已修改完毕 (y/n): " confirm

if [ "$confirm" == "y" ]; then
    # 运行 Python 脚本
    python -m head.scripts.main_head --train_flag=1 --total_steps=1000000 --train_name=experiment_1
else
    echo "请先修改配置文件，然后重新运行脚本。"
fi