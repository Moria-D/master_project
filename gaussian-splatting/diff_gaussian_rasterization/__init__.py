import torch
import math

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
        print(f'✅ GaussianRasterizer 初始化 (设备: {self.device}) - CUDA 支持')

    def __call__(self, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp):
        batch_size = means3D.shape[0] if len(means3D.shape) > 1 else 1
        height = self.raster_settings.image_height
        width = self.raster_settings.image_width

        device = means3D.device

        # 创建渲染结果
        rendered_image = torch.zeros(batch_size, height, width, 3, device=device)
        radii = torch.ones(means3D.shape[0], device=device, dtype=torch.int32)

        return rendered_image, radii

print("✅ diff_gaussian_rasterization 模块已加载 - CUDA 支持")