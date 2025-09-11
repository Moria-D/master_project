#!/usr/bin/env python3
"""
测试3DGS模块导入的脚本
"""
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
gaussian_splatting_path = os.path.join(project_root, 'gaussian-splatting')
sys.path.insert(0, gaussian_splatting_path)

print("=== 3DGS 模块测试 ===")
print(f"Python 版本: {sys.version}")
print(f"项目路径: {project_root}")
print(f"gaussian-splatting 路径: {gaussian_splatting_path}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} 可用")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"❌ PyTorch 不可用: {e}")
    sys.exit(1)

try:
    import simple_knn
    print("✅ simple_knn 可用")
except ImportError as e:
    print(f"❌ simple_knn 不可用: {e}")

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print("✅ diff_gaussian_rasterization 可用")
except ImportError as e:
    print(f"❌ diff_gaussian_rasterization 不可用: {e}")
    import traceback
    traceback.print_exc()

try:
    from gaussian_renderer import render, network_gui
    print("✅ gaussian_renderer 可用")
except ImportError as e:
    print(f"❌ gaussian_renderer 不可用: {e}")
    import traceback
    traceback.print_exc()

try:
    from scene.gaussian_model import GaussianModel
    print("✅ GaussianModel 可用")
except ImportError as e:
    print(f"❌ GaussianModel 不可用: {e}")
    import traceback
    traceback.print_exc()

try:
    from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
    print("✅ utils.general_utils 可用")
except ImportError as e:
    print(f"❌ utils.general_utils 不可用: {e}")

try:
    from utils.sh_utils import eval_sh, RGB2SH
    print("✅ utils.sh_utils 可用")
except ImportError as e:
    print(f"❌ utils.sh_utils 不可用: {e}")

print("\n=== 模块测试完成 ===")

# 如果所有模块都可用，打印成功信息
success = True
try:
    import torch
    import simple_knn
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import inverse_sigmoid
    from utils.sh_utils import eval_sh
except ImportError:
    success = False

if success:
    print("🎉 所有模块都可用！可以开始运行 3DGS 实验了！")
    print("\n运行命令：")
    print("export PYTHONPATH=\"$PWD/gaussian-splatting:\$PYTHONPATH\"")
    print("./quick_test.sh room 0")
    print("./run_ntu_experiment.sh room 0")
else:
    print("❌ 部分模块不可用，需要进一步修复")
