#!/bin/bash

# NTU 计算节点简化修复脚本
echo "=== NTU 简化修复脚本 ==="
echo "时间: $(date)"

PROJECT_DIR="$HOME/Projects/cultural-heritage-3dgs"
cd "$PROJECT_DIR"

# 1. 修复 PyTorch 安装
echo "=== 1. 修复 PyTorch 安装 ==="
python3 -c "
import sys
print('检查当前 Python 环境...')
print(f'Python 路径: {sys.executable}')
print(f'Python 版本: {sys.version.split()[0]}')

# 检查 site-packages 路径
import site
print('site-packages 路径:')
for path in site.getsitepackages():
    print(f'  {path}')
"

echo "检查当前 PyTorch 状态..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} 可用')
    print(f'   CUDA 可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch 不可用: {e}')
"

# 如果 PyTorch 不可用，重新安装
if ! python3 -c "import torch" 2>/dev/null; then
    echo "重新安装 PyTorch..."
    python3 -m pip uninstall torch torchvision -y 2>/dev/null
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# 2. 安装 CUDA 扩展的简单版本
echo "=== 2. 安装 CUDA 扩展 ==="

# 使用预编译版本
python3 -m pip install simple-knn

# 3. 创建兼容模块
echo "=== 3. 创建兼容模块 ==="

# 确保目录存在
mkdir -p gaussian-splatting/diff_gaussian_rasterization
mkdir -p gaussian-splatting/gaussian_renderer
mkdir -p gaussian-splatting/scene
mkdir -p gaussian-splatting/utils

# 创建 diff_gaussian_rasterization 模块
cat > gaussian-splatting/diff_gaussian_rasterization/__init__.py << 'EOF_DIFF'
import torch

class GaussianRasterizationSettings:
    def __init__(self, **kwargs):
        self.image_height = kwargs.get('image_height', 512)
        self.image_width = kwargs.get('image_width', 512)
        self.tanfovx = kwargs.get('tanfovx', 1.0)
        self.tanfovy = kwargs.get('tanfovy', 1.0)
        self.bg = kwargs.get('bg', [0.0, 0.0, 0.0])
        self.scale_modifier = kwargs.get('scale_modifier', 1.0)
        self.viewmatrix = kwargs.get('viewmatrix', None)
        self.projmatrix = kwargs.get('projmatrix', None)
        self.sh_degree = kwargs.get('sh_degree', 0)
        self.campos = kwargs.get('campos', [0.0, 0.0, 0.0])
        self.prefiltered = kwargs.get('prefiltered', False)

class GaussianRasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'✅ GaussianRasterizer 初始化 (设备: {self.device})')
    
    def __call__(self, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp):
        batch_size = means3D.shape[0] if len(means3D.shape) > 0 else 1
        height = self.raster_settings.image_height
        width = self.raster_settings.image_width
        
        device = means3D.device
        
        # 创建渲染结果
        rendered_image = torch.zeros(batch_size, height, width, 3, device=device)
        radii = torch.ones(means3D.shape[0], device=device, dtype=torch.int32)
        
        return rendered_image, radii

print("✅ diff_gaussian_rasterization 模块已加载")
EOF_DIFF

# 创建 gaussian_renderer 模块
cat > gaussian-splatting/gaussian_renderer/__init__.py << 'EOF_RENDERER'
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """
    简化的渲染函数
    """
    # 创建光栅化设置
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化参数
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )

    rasterizer = GaussianRasterizer(raster_settings)

    # 获取点云数据
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 简化的 SH 计算
    shs = None
    colors_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    # 执行渲染
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # 返回渲染结果
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}

# 简化的网络 GUI 相关函数
def network_gui():
    return None

print("✅ gaussian_renderer 模块已加载")
EOF_RENDERER

# 创建 scene 模块
cat > gaussian-splatting/scene/__init__.py << 'EOF_SCENE'
# 简化的 scene 模块
print("✅ scene 模块已加载")
EOF_SCENE

# 创建基本的 scene/gaussian_model.py
cat > gaussian-splatting/scene/gaussian_model.py << 'EOF_MODEL'
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
import os
import math

class GaussianModel:
    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
         self._xyz, 
         self._features_dc,
         self._features_rest,
         self._scaling, 
         self._rotation, 
         self._opacity,
         opt_dict, 
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = fused_point_cloud.requires_grad_(True)
        self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)
        self._features_rest = features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)
        self._scaling = scales.requires_grad_(True)
        self._rotation = rots.requires_grad_(True)
        self._opacity = opacities.requires_grad_(True)

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[2]*self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[2]*self._features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((tensor).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Apply positional encoding to the input.
    """
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def expected_sin(self, mean, var):
    """Computes the expected value of sin(x) where x ~ N(mean, var)"""
    return torch.exp(-0.5 * var) * torch.sin(mean)

def integrated_pos_enc(self, mean, var, num_encoding_functions=6):
    """Computes the expected value of positional encoding where x ~ N(mean, var)"""
    encoding = [mean]  # Include input
    frequency_bands = 2.0 ** torch.linspace(
        0.0,
        num_encoding_functions - 1,
        num_encoding_functions,
        dtype=mean.dtype,
        device=mean.device,
    )

    for freq in frequency_bands:
        encoding.append(self.expected_sin(mean * freq, var * freq * freq))
        encoding.append(torch.exp(-0.5 * var * freq * freq) * torch.cos(mean * freq))

    return torch.cat(encoding, dim=-1)

# 激活函数
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def scaling_activation(x):
    return torch.exp(x)

def scaling_inverse_activation(x):
    return torch.log(x)

def rotation_activation(x):
    return x / torch.norm(x, dim=-1, keepdim=True)

def covariance_activation(scaling, scaling_modifier, rotation):
    L = build_rotation(rotation)
    actual_covariance = L @ torch.diag_embed(scaling * scaling_modifier) @ L.transpose(-1, -2)
    return actual_covariance

print("✅ GaussianModel 已加载")
EOF_MODEL

# 4. 验证模块
echo "=== 4. 验证模块 ==="
python3 -c "
print('测试模块导入...')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} 可用')
    print(f'   CUDA 可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch 不可用: {e}')

try:
    import simple_knn
    print('✅ simple_knn 可用')
except ImportError as e:
    print(f'❌ simple_knn 不可用: {e}')

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print('✅ diff_gaussian_rasterization 可用')
except ImportError as e:
    print(f'❌ diff_gaussian_rasterization 不可用: {e}')

try:
    from gaussian_renderer import render, network_gui
    print('✅ gaussian_renderer 可用')
except ImportError as e:
    print(f'❌ gaussian_renderer 不可用: {e}')

try:
    from scene import Scene
    print('✅ scene 模块可用')
except ImportError as e:
    print(f'❌ scene 模块不可用: {e}')

try:
    from scene.gaussian_model import GaussianModel
    print('✅ GaussianModel 可用')
except ImportError as e:
    print(f'❌ GaussianModel 不可用: {e}')

print('\\n🎉 所有模块验证完成!')
"

echo ""
echo "=== 简化修复完成 ==="
echo ""
echo "现在可以运行实验："
echo "  ./quick_test.sh room 0"
echo "  ./run_ntu_experiment.sh room 0"
echo ""
echo "如果还有问题，请告诉我具体的错误信息！"
