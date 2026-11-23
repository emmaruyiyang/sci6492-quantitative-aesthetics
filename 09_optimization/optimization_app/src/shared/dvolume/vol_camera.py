import math

from .vol_bounds import *
from .vol import *
from .vol_transform3d import *


class VolumeCamera:

    @staticmethod
    def makeWithAngles(
        distance : float,
        angle_xy_deg : float, 
        angle_z_deg : float,
        target_point : Vec3 = [0.0, 0.0, 0.0],
        up_vector : Vec3 = [0.0, 0.0, 1.0],
        fov_deg : float = 50.0,
        aspect : float = 1.0,
        znear : float = 0.01,
        zfar : float = 10.0,
        device : torch.device = Vol.device,
        orthographic : bool = False,
        ortho_half_width : float = 3.0
        ) -> 'VolumeCamera':

        angle_xy = angle_xy_deg * math.pi / 180.0
        angle_z = angle_z_deg * math.pi / 180.0

        view_direction = [
            distance * math.cos(angle_xy) * math.cos(angle_z),
            distance * math.sin(angle_xy) * math.cos(angle_z),
            distance * math.sin(angle_z)
        ]

        view_direction = ensureTensor(view_direction, device)
        target_point = ensureTensor(target_point, device)

        view_point = target_point + view_direction

        return VolumeCamera(view_point, target_point, up_vector, fov_deg, aspect, znear, zfar, device, orthographic, ortho_half_width)
        

    def __init__(self, 
                 view_point : Vec3, 
                 target_point : Vec3, 
                 up_vector : Vec3,
                 fov_deg : float, 
                 aspect : float, 
                 znear : float, 
                 zfar : float,
                 device : torch.device = Vol.device,
                orthographic : bool = False,
                ortho_half_width : float = 3.0
                 ):
        
        self.device = device

        self.view_point = ensureTensor(view_point, self.device)
        self.target_point = ensureTensor(target_point, self.device)
        self.up_vector = ensureTensor(up_vector, self.device)

        self.fov_deg = fov_deg
        self.aspect = aspect
        self.znear = znear
        self.zfar = zfar

        self.orthographic = orthographic
        self.ortho_half_width = ortho_half_width

        self.update()       

    @property
    def distance(self) -> float:
        return torch.norm(self.view_point - self.target_point).item()   
    
    @property
    def direction(self) -> torch.Tensor:
        dir = self.view_point - self.target_point
        dir = dir / torch.norm(dir)
        return dir
    
    @property
    def angle_xy_deg(self) -> float:
        dir = self.direction
        angle_xy = math.atan2(dir[1].item(), dir[0].item())
        return angle_xy * 180.0 / math.pi
    
    @property
    def angle_z_deg(self) -> float:
        dir = self.direction
        angle_z = math.asin(dir[2].item())
        return angle_z * 180.0 / math.pi

    def setWithAngles(self,
                        distance : float,
                        angle_xy_deg : float,
                        angle_z_deg : float,
                        target_point : Vec3 = None,
                        up_vector : Vec3 = None,
                        fov_deg : float = None,
                        aspect : float = None,
                        znear : float = None,
                        zfar : float = None,
                        orthographic : bool = None,
                        ortho_half_width : float = None
                        ):
        angle_xy = angle_xy_deg * math.pi / 180.0
        angle_z = angle_z_deg * math.pi / 180.0

        view_direction = [
            distance * math.cos(angle_xy) * math.cos(angle_z),
            distance * math.sin(angle_xy) * math.cos(angle_z),
            distance * math.sin(angle_z)
        ]

        view_direction = ensureTensor(view_direction, self.device)
        target_point = target_point or self.target_point
        target_point = ensureTensor(target_point, self.device)
        view_point = target_point + view_direction

        orthographic = orthographic if orthographic is not None else self.orthographic
        ortho_half_width = ortho_half_width if ortho_half_width is not None else self.ortho_half_width

        self.set(view_point, target_point, up_vector, fov_deg, aspect, znear, zfar, orthographic, ortho_half_width)
    

    def set(self,
                        view_point : None,
                        target_point : None,
                        up_vector : None,
                        fov_deg : float = None,
                        aspect : float = None,
                        znear : float = None,
                        zfar : float = None,
                        orthographic : bool = None,
                        ortho_half_width : float = None
                        ):
        
        if view_point is not None:
            self.view_point = ensureTensor(view_point, self.device)
        
        if target_point is not None:
            self.target_point = ensureTensor(target_point, self.device)
        
        if up_vector is not None:
            self.up_vector = ensureTensor(up_vector, self.device)

        self.fov_deg = fov_deg or self.fov_deg
        self.aspect = aspect or self.aspect
        self.znear = znear or self.znear
        self.zfar = zfar or self.zfar

        self.orthographic = orthographic if orthographic is not None else self.orthographic
        self.ortho_half_width = ortho_half_width if ortho_half_width is not None else self.ortho_half_width

    def update(self):       
        self.view_matrix = Transform3d.makeLookAt(self.view_point, self.target_point, self.up_vector, self.device)
        if self.orthographic:
            self.projection_matrix = Transform3d.makeOrthographic(-self.ortho_half_width, self.ortho_half_width,
                                                                   self.ortho_half_width/self.aspect, -self.ortho_half_width/self.aspect,
                                                                   self.znear, self.zfar, self.device)
        else:
            self.projection_matrix = Transform3d.makePerspective(self.fov_deg, self.aspect, self.znear, self.zfar, True, self.device)

        self.view_proj_matrix = self.projection_matrix @ self.view_matrix
        self.inv_view_proj_matrix = self.view_proj_matrix.getInverse()
        

    def projectPoints(self, points_3d : torch.Tensor, divideHomogeneous : bool = True) -> torch.Tensor:
        return self.view_proj_matrix.transformPoints(points_3d, divideHomogeneous)
        
    def unprojectPoints(self, points_2d : torch.Tensor, divideHomogeneous : bool = True) -> torch.Tensor:
        return self.inv_view_proj_matrix.transformPoints(points_2d, divideHomogeneous)
      
    def createViewSamplingGrid(self, 
                           resolution : list[int], #depth, height, width
                           view_min : list[float], #min x, min y, min z
                           view_max : list[float], #max x, max y, max z
                           depth_noise : float,
                           model_matrix : Transform3d = None
                           ) -> torch.Tensor:
        
        nz = resolution[0]
        ny = resolution[1]
        nx = resolution[2]

        z_min = view_min[2]
        z_max = view_max[2]

        x = torch.linspace(view_min[0], view_max[0], nx, dtype=torch.float32, device=self.device)
        y = torch.linspace(view_min[1], view_max[1], ny, dtype=torch.float32, device=self.device)
        z = torch.linspace(view_min[2], view_max[2], nz, dtype=torch.float32, device=self.device)

        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        x = x.unsqueeze(3)
        y = y.unsqueeze(3)
        z = z.unsqueeze(3)

        if depth_noise > 0.0:
            z = z + torch.randn_like(z, dtype=torch.float32, device=self.device) * (depth_noise*(z_max-z_min)/nz)

        #viewport grid points [-1, 1]x[-1, 1]x[min_depth, max_depth] in desired image resolution
        grid_viewport = torch.cat([x, y, z, torch.ones_like(x)], dim=3)
        #set order to Depth, Height, Width, Channels
        grid_viewport = grid_viewport.permute(2, 1, 0, 3)

        return self.createSamplingGridFromViewportGrid(grid_viewport, model_matrix)
    
    def createSamplingGridFromViewportGrid(self, grid_viewport : torch.Tensor, model_matrix : Transform3d = None) -> torch.Tensor:
        unproject_matrix = self.inv_view_proj_matrix

        if model_matrix is not None:
            inv_model_matrix = model_matrix.getInverse()
            unproject_matrix = inv_model_matrix @ unproject_matrix

        #transform grid to volume space using inverse perspective matrix
        grid_volume = torch.matmul(grid_viewport, unproject_matrix.tensor.t())
        grid_volume = grid_volume / grid_volume[:, :, :, 3].unsqueeze(3)
        grid_volume = grid_volume[:, :, :, 0:3]
        grid_volume = grid_volume.unsqueeze(0)

        return grid_volume
    
    def getViewProjectedBounds(self, 
                               world_bounds:Bounds3d = Bounds3d.makeUnitSymmetricCube(),
                               model_matrix:Transform3d = None
                               ) -> Bounds3d:
        corners_3d = world_bounds.getCornersTensor(device=self.device)
        if model_matrix is not None:
            corners_3d = model_matrix.transformPoints(corners_3d)
        corners_2d = self.projectPoints(corners_3d)
        return Bounds3d.makeFromPointsTensor(corners_2d)
    
    def createDebugCubeCorners(self, 
                               image_resolution:int,                                
                               world_bounds:Bounds3d = Bounds3d.makeUnitSymmetricCube(),
                               model_matrix:Transform3d = None
                               ) -> torch.Tensor:

        #transform the corners of the volume to 2d image space
        corners_3d = world_bounds.getCornersTensor(self.device)
        if model_matrix is not None:
            corners_3d = model_matrix.transformPoints(corners_3d)
        corners_2d = self.projectPoints(corners_3d)

        canvas = torch.ones(1, 3, image_resolution, image_resolution) * 0.5
        #draw the 2d corners on the canvas
        for i in range(8):
            x = int((corners_2d[i, 0].item()*0.5 + 0.5)*image_resolution)
            y = int((corners_2d[i, 1].item()*0.5 + 0.5)*image_resolution)
            if x >= 0 and x < image_resolution and y >= 0 and y < image_resolution:
                xyz = corners_3d[i, :3]*0.5 + 0.5
                canvas[0, :, y, x] = xyz
        
        return canvas