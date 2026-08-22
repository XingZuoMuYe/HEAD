import os

# Keep tests independent from ROS/system CUDA libraries on developer machines.
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_HOME", None)
