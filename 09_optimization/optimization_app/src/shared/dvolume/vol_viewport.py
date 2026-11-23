import torch
import torchvision.transforms.functional as VF

from .vol_types import *
from .vol_camera import *
from .vol_model import *
from .vol_lighting import *

from enum import Enum
from typing import Callable, Optional, Union

class VolumeRendererCompositing(Enum):
    ALPHA = 1
    ADD = 2
    MAX = 3

    @staticmethod
    def __alpha(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst * (1.0 - alpha) + src * alpha
    
    @staticmethod
    def __add(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst + src * alpha
    
    @staticmethod
    def __max(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return torch.max(dst, src*alpha)
    
    @staticmethod
    def __alpha_a(dst_alpha : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst_alpha * (1.0 - alpha) + alpha
    
    @staticmethod
    def __add_a(dst_alpha : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst_alpha + alpha
    
    @staticmethod
    def __max_a(dst_alpha : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return torch.max(dst_alpha, alpha)
    
    @staticmethod
    def __alpha_premult(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst * (1.0 - alpha) + src
    
    @staticmethod
    def __add_premult(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return dst + src
    
    @staticmethod
    def __max_premult(dst : torch.Tensor, src : torch.Tensor, alpha : torch.Tensor) -> torch.Tensor:
        return torch.max(dst, src)
    
    def getColorCompositor(compositing : 'VolumeRendererCompositing') -> Callable:
        if compositing == VolumeRendererCompositing.ALPHA:
            return VolumeRendererCompositing.__alpha
        elif compositing == VolumeRendererCompositing.ADD:
            return VolumeRendererCompositing.__add
        elif compositing == VolumeRendererCompositing.MAX:
            return VolumeRendererCompositing.__max
        else:
            raise ValueError('Invalid compositing mode')
        
    def getAlphaCompositor(compositing : 'VolumeRendererCompositing') -> Callable:
        if compositing == VolumeRendererCompositing.ALPHA:
            return VolumeRendererCompositing.__alpha_a
        elif compositing == VolumeRendererCompositing.ADD:
            return VolumeRendererCompositing.__add_a
        elif compositing == VolumeRendererCompositing.MAX:
            return VolumeRendererCompositing.__max_a
        else:
            raise ValueError('Invalid compositing mode')
        
    def getColorCompositorPremult(compositing : 'VolumeRendererCompositing') -> Callable:
        if compositing == VolumeRendererCompositing.ALPHA:
            return VolumeRendererCompositing.__alpha_premult
        elif compositing == VolumeRendererCompositing.ADD:
            return VolumeRendererCompositing.__add_premult
        elif compositing == VolumeRendererCompositing.MAX:
            return VolumeRendererCompositing.__max_premult
        else:
            raise ValueError('Invalid compositing mode')
    

class GBuffers :
    def __init__(self, 
                 width : int, height : int,
                 density :  torch.Tensor = None, 
                 position : torch.Tensor = None, 
                 normals : torch.Tensor = None, 
                 alpha : torch.Tensor = None,
                 color : torch.Tensor = None,
                 diffuse : torch.Tensor = None,
                 specular : torch.Tensor = None,
                 final_color : torch.Tensor = None
                 ):
        self.width = width
        self.height = height
        self.density = density
        self.position = position
        self.normals = normals

        self.alpha = alpha
        self.color = color

        self.diffuse = diffuse
        self.specular = specular

        self.final_color = final_color

class VolumeViewport:
    DEFAULT_DEPTH_LAYERS = 60
    DEFAULT_OPACITY = 0.8
    DEFAULT_SIGMOID_FACTOR = 60.0
    DEFAULT_NOISE = 0.15

    def __init__(self, 
                 camera : VolumeCamera, 
                 width : int,
                 height : int, 
                 depth_layers : int = DEFAULT_DEPTH_LAYERS,
                 depth_noise : float = DEFAULT_NOISE
                 ):
        
        self.__resolution = [depth_layers, height, width]
        self.camera = camera
        self.depth_noise = depth_noise

        self.min_depth = 0.0
        self.max_depth = 1.0
        self.__grid_viewport : torch.Tensor = None        
        
    
    @property
    def depth_layers(self):
        return self.__resolution[0]
    
    @property
    def resolution(self):
        return self.__resolution
    
    @property
    def device(self):
        return self.camera.device
    
    @property
    def width(self):
        return self.__resolution[2]
    
    @property
    def height(self):
        return self.__resolution[1]
    
    @property
    def depth_layers(self):
        return self.__resolution[0]
    
    def ensureGridViewportForVolume(self, volume : VolumeModel) -> torch.Tensor:
        bounds_2d = volume.getViewBounds(self.camera)
        return self.ensureGridViewport(bounds_2d.min_z, bounds_2d.max_z)

    def ensureGridViewport(self, min_depth : float, max_depth : float) -> torch.Tensor:
        if self.__grid_viewport is not None and self.min_depth == min_depth and self.max_depth == max_depth:
            return self.__grid_viewport
        
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.__x = torch.linspace(-1.0, 1.0, self.width, dtype=torch.float32, device=self.device)
        self.__y = torch.linspace(-1.0, 1.0, self.height, dtype=torch.float32, device=self.device)
        self.__z = torch.linspace(min_depth, max_depth, self.depth_layers, dtype=torch.float32, device=self.device)
        self.__x, self.__y, self.__z = torch.meshgrid(self.__x, self.__y, self.__z, indexing='ij')

        self.__x = self.__x.unsqueeze(3)
        self.__y = self.__y.unsqueeze(3)
        self.__z = self.__z.unsqueeze(3)

        if self.depth_noise > 0.0:
            noise_scale = self.depth_noise*(max_depth-min_depth)/self.depth_layers
            self.__z = self.__z + torch.randn_like(self.__z, dtype=torch.float32, device=self.device) * noise_scale

        self.__w = torch.ones_like(self.__x, dtype=torch.float32, device=self.device)

        #viewport grid points [-1, 1]x[-1, 1]x[min_depth, max_depth] in desired image resolution
        self.__grid_viewport = torch.cat([self.__x, self.__y, self.__z, self.__w], dim=3)
        #set order to Depth, Height, Width, Channels
        self.__grid_viewport = self.__grid_viewport.permute(2, 1, 0, 3)

        return self.__grid_viewport
    
    # def createCompatibleCanvasImage(self,  background_color : Vec3) -> torch.Tensor:
    #     canvas = torch.zeros(3, self.height, self.width, dtype=torch.float32, device=self.device)
    #     canvas[0, :, :] = background_color[0]
    #     canvas[1, :, :] = background_color[1]
    #     canvas[2, :, :] = background_color[2]
    #     return canvas
    
    def createCompatibleCanvasImage(self, fill_value: Union[torch.Tensor, float, list[float]]) -> torch.Tensor:
        fill_value = ensureTensor(fill_value, self.device)
        channels = fill_value.shape[0]
        canvas = self.__createCompatibleCanvasImageZeros(channels)

        for i in range(channels):
            canvas[i, :, :] = fill_value[i]
        
        return canvas
    
    def __createCompatibleCanvasImageZeros(self,  channels:int) -> torch.Tensor:
        return torch.zeros(channels, self.height, self.width, dtype=torch.float32, device=self.device)

    def createSamplingGrid(self, 
                     volume : VolumeModel,        
                     recompute_depth_bounds : bool = True
                     ) -> torch.Tensor:
        
        if recompute_depth_bounds or self.__grid_viewport is None:
            bounds_2d = volume.getViewBounds(self.camera)
            grid_viewport = self.ensureGridViewport(bounds_2d.min_z, bounds_2d.max_z)
        else:
            grid_viewport = self.__grid_viewport

        return self.camera.createSamplingGridFromViewportGrid(grid_viewport, volume.model_matrix)

    def __createBackgroundImage(self, background : Union[Vec3, torch.Tensor]) -> torch.Tensor:
        if isinstance(background, torch.Tensor) and len(background.shape) == 3:
            if background.shape[0] == 3:    
                w = background.shape[2]
                h = background.shape[1]
                if w == self.width and h == self.height:
                    return background
                else:
                    return VF.resize(background, (self.height, self.width))
            else:
                raise ValueError('Invalid background image shape, requires 3 channels')
        else:
            return self.createCompatibleCanvasImage(background)
        
    def renderVolume(self, 
                     volume : VolumeModel,     
                     background : Union[Vec3, torch.Tensor],   #{3} | {3, height, width}                     
                     *,          
                     volume_color : Optional[Vec3] = None,                #{3}              
                     compositing_opacity : float = DEFAULT_OPACITY,
                     density_sigmoid_factor : float = DEFAULT_SIGMOID_FACTOR,
                     lighting : Optional[VolLighting] = None,
                     recompute_depth_bounds : bool = True,
                     volume_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                     alpha_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                     require_alpha : bool = False,
                     fog_color : Optional[Vec3] = None,                                       
                     ) -> GBuffers:
        
        gb = GBuffers(self.width, self.height)

        comp_color_fn = VolumeRendererCompositing.getColorCompositor(volume_compositing)
        comp_alpha_fn = VolumeRendererCompositing.getAlphaCompositor(alpha_compositing)
        comp_color_premult_fn = VolumeRendererCompositing.getColorCompositorPremult(volume_compositing)

        #get the view frustrum grid for the volume, the voxels in this grid
        #hold the sampling UVW positions from the volume model so that it is 
        #projected to the viewport
        #{1, layers, view_height, view_width, 3}
        sampling_grid = self.createSamplingGrid(volume, recompute_depth_bounds)

        #get the color field for the volume
        if lighting is not None:
            #{1, 3, Vol_D, Vol_H, Vol_W}
            color = volume.getLitColorField(lighting)
        else:
            color = volume.color
       
        #sample the volume density and color using to the view frustrum grid
        #{1, Channels, layers, view_height, view_width}
        density_view = volume.density.resample(sampling_grid)
        color_view = color.resample(sampling_grid) if color is not None else None

        #create 2d buffers for final color and optionally alpha
        #{3, view_height, view_width}
        gb.final_color = self.__createBackgroundImage(background)
        if require_alpha:
            #{1, view_height, view_width}
            gb.alpha = self.createCompatibleCanvasImage(0.0)

        #compute the alpha tensor from the volume density
        #the sigmoid function creates a sharp transition from solid to void where the density is 0.5
        if density_sigmoid_factor>0.0:
            #{1, 1, layers, view_height, view_width}
            alpha_tensor = torch.sigmoid((density_view.tensor-0.5)*density_sigmoid_factor) * compositing_opacity
        else:
            #{1, 1, layers, view_height, view_width}
            alpha_tensor = density_view.tensor * compositing_opacity
    
        #compute color compositing
        if color_view is not None:
            color_tensor = color_view.tensor * alpha_tensor

            if volume_color is not None:
                volume_color = ensureTensor(volume_color, self.device)
                volume_color = volume_color.reshape(1, 3, 1, 1, 1)
                color_tensor = color_tensor * volume_color

            if fog_color is not None:
                fog_color = ensureTensor(fog_color, self.device)

            for i in range(self.depth_layers-1, -1, -1):
                slice_alpha = alpha_tensor[0, :, i]
                slice_color = color_tensor[0, :, i]


                # if fog_color is not None:
                #     fog_density = 1.0 - (i / self.depth_layers)
                #     slice_color = slice_color * fog_density + fog_color * (1.0 - fog_density)

                
                #gb.final_color = gb.final_color * (1.0 - slice_alpha) + slice_color
                gb.final_color = comp_color_premult_fn(gb.final_color, slice_color, slice_alpha)

                if require_alpha:
                    gb.alpha = comp_alpha_fn(gb.alpha, slice_alpha)
        else:
            if volume_color is None:
                volume_color = [1.0, 1.0, 1.0]

            if fog_color is not None:
                fog_color = ensureTensor(fog_color, self.device)
                fog_color = fog_color.reshape(3, 1, 1)

            volume_color = ensureTensor(volume_color, self.device)
            slice_color = volume_color.reshape(3, 1, 1)

           


            for i in range(self.depth_layers-1, -1, -1):
                slice_alpha = alpha_tensor[0, :, i]

                if fog_color is not None:
                    fog_density = 1.0 - (i / self.depth_layers)
                    slice_color_fog = slice_color * fog_density + fog_color * (1.0 - fog_density)
                    gb.final_color = comp_color_fn(gb.final_color, slice_color_fog, slice_alpha)
                else:
                    #gb.final_color = gb.final_color * (1.0 - slice_alpha) + slice_color * slice_alpha
                    gb.final_color = comp_color_fn(gb.final_color, slice_color, slice_alpha)

                if require_alpha:
                    gb.alpha = comp_alpha_fn(gb.alpha, slice_alpha)


        return gb
    
    def createGBuffers(self, 
                     volume : VolumeModel,                     
                     *,          
                     volume_color : Optional[Vec3] = None,                #{3}              
                     compositing_opacity : float = DEFAULT_OPACITY,
                     density_sigmoid_factor : float = DEFAULT_SIGMOID_FACTOR,
                     recompute_depth_bounds : bool = True,
                     volume_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                     alpha_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                     require_position : bool = True,
                     require_normals : bool = True,
                     require_color : bool = True
                     ) -> GBuffers:
              

        gb = GBuffers(self.width, self.height)

        comp_color_fn = VolumeRendererCompositing.getColorCompositor(volume_compositing)
        comp_alpha_fn = VolumeRendererCompositing.getAlphaCompositor(alpha_compositing)
        comp_color_premult_fn = VolumeRendererCompositing.getColorCompositorPremult(volume_compositing)


        #{1, layers, view_height, view_width, 3}
        sampling_grid = self.createSamplingGrid(volume, recompute_depth_bounds)

        density_view : VolumeField = None
        color_view : VolumeField = None
        normals_view : VolumeField = None
        points_view : VolumeField = None
        
        #{1, 1, depth, height, width} -> {1, 1, layers, view_height, view_width}
        density_view = volume.density.resample(sampling_grid).tensor
        #{1, view_height, view_width}
        gb.density = self.__createCompatibleCanvasImageZeros(1)
        
        #{1, 1, layers, view_height, view_width}
        if density_sigmoid_factor>0.0:
            alpha_tensor = torch.sigmoid((density_view-0.5)*density_sigmoid_factor) * compositing_opacity
        else:
            alpha_tensor = density_view * compositing_opacity
        
        #{1, 3, depth, height, width} -> {1, 3, layers, view_height, view_width}
        if require_color and volume.color is not None:            
            color_view = volume.color.resample(sampling_grid).tensor

        #{1, 3, depth, height, width} -> {1, 3, layers, view_height, view_width}
        if require_normals:
            normals_view = volume.ensureNormals().resample(sampling_grid).tensor * alpha_tensor
            #{3, view_height, view_width}
            gb.normals = self.__createCompatibleCanvasImageZeros(3) 

        if require_position:
            #{depth, height, width, 4}
            points = volume.uv_coords.xyzw
            #{1, 3, depth, height, width}
            points = VolumeField(points[:,:,:,0:3].permute(3, 0, 1, 2).unsqueeze(0))
            
            #{1, 3, layers, view_height, view_width}
            points_view = points.resample(sampling_grid).tensor * alpha_tensor   
                
            #{3, view_height, view_width}
            gb.position = self.__createCompatibleCanvasImageZeros(3)


        for i in range(self.depth_layers-1, -1, -1):
            slice_alpha = alpha_tensor[0, :, i]

            #gb.density = gb.density * (1.0 - slice_alpha) + slice_alpha
            gb.density = comp_alpha_fn(gb.density, slice_alpha)
            
            if require_normals:
                slice_normal = normals_view[0, :, i]                
                #gb.normals = gb.normals * (1.0 - slice_alpha) + slice_normal * slice_alpha
                gb.normals = comp_color_premult_fn(gb.normals, slice_normal, slice_alpha)
            
            if require_position:
                slice_position = points_view[0, :, i]
                #gb.position = gb.position * (1.0 - slice_alpha) + slice_position * slice_alpha
                gb.position = comp_color_premult_fn(gb.position, slice_position, slice_alpha)
            
        if require_color:
            if color_view is not None:
                #{3, view_height, view_width}
                gb.color = self.__createCompatibleCanvasImageZeros(3)
                color_tensor = color_view * alpha_tensor

                if volume_color is not None:
                    volume_color = ensureTensor(volume_color, self.device).reshape(1, 3, 1, 1, 1)
                    color_tensor = color_tensor * volume_color

                

                for i in range(self.depth_layers-1, -1, -1):
                    slice_alpha = alpha_tensor[0, :, i]
                    slice_color = color_tensor[0, :, i]
                    #gb.color = gb.color * (1.0 - slice_alpha) + slice_color
                    gb.color = comp_color_premult_fn(gb.color, slice_color, slice_alpha)
            else:
                if volume_color is None:
                    volume_color = [1.0, 1.0, 1.0]

                volume_color = ensureTensor(volume_color, self.device).reshape(3, 1, 1)
                gb.color = gb.density * volume_color

        gb.alpha = torch.clamp(gb.density, 0.0, 1.0)

        return gb
    
    def renderVolumeUsingGBuffers(self, 
                                    volume : VolumeModel,  
                                    background : Union[Vec3, torch.Tensor],   #{3} | {3, height, width}                   
                                    *,          
                                    volume_color : Optional[Vec3] = None,                #{3}              
                                    compositing_opacity : float = DEFAULT_OPACITY,
                                    density_sigmoid_factor : float = DEFAULT_SIGMOID_FACTOR,
                                    lighting : Optional[VolLighting] = None,
                                    recompute_depth_bounds : bool = True,
                                    volume_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                                    alpha_compositing : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                                    require_position : bool = True,
                                    require_normals : bool = True,
                                    fog_color : Optional[Vec3] = None,
                                ) -> GBuffers:
        
        if lighting is not None:
            require_normals = True
            if lighting.isPointLight:
                require_position = True

        gb = self.createGBuffers(
                volume, 
                volume_color =              volume_color, 
                compositing_opacity =       compositing_opacity, 
                density_sigmoid_factor =    density_sigmoid_factor, 
                recompute_depth_bounds =    recompute_depth_bounds, 
                volume_compositing =        volume_compositing,
                alpha_compositing =         alpha_compositing,
                require_position =          require_position, 
                require_normals=            require_normals,
                require_color =             True
            )
        

        gb.final_color = self.__createBackgroundImage(background)

        if lighting is None:
            gb.final_color = gb.final_color*(1.0 - gb.alpha) + gb.color*gb.alpha
        else:
            light = lighting.light_dir.reshape(3, 1, 1)    

            if lighting.isPointLight:
                camera_pos = self.camera.view_point.reshape(3, 1, 1)    
                spec_color = ensureTensor([1.0, 1.0, 1.0], self.device).reshape(3, 1, 1)

                #blinn-phong
                light_dir = (light - gb.position)
                light_dir = light_dir / torch.norm(light_dir, dim=0)
                view_dir = camera_pos - gb.position
                view_dir = view_dir / torch.norm(view_dir, dim=0)
                half_dir = (light_dir + view_dir)
                half_dir = half_dir / torch.norm(half_dir, dim=0)

                #gb.diffuse = torch.sum(gb.normals * light_dir, dim=0).clamp(0.0, 1.0)
                gb.diffuse = (torch.sum(gb.normals * light_dir, dim=0)*0.5 + 0.5).clamp(0.0, 1.0)
                gb.specular = torch.sum(gb.normals * half_dir, dim=0).clamp(0.0, 1.0)
                gb.specular = torch.pow(gb.specular, lighting.specular_power).clamp(0.0, 1.0)

                diff_color = (gb.color*gb.diffuse) + (spec_color*gb.specular) 

                if lighting.ambient_color is not None:
                    ambient = lighting.ambient_color.reshape(3, 1, 1)
                    diff_color = diff_color + ambient

                gb.final_color = gb.final_color*(1.0 - gb.alpha) + diff_color*gb.alpha
            else:                
                gb.diffuse = (torch.sum(gb.normal * light, dim=0)*0.5+0.5).clamp(0.0, 1.0)  
                #gb.diffuse = (torch.sum(gb.normals * light, dim=0)).clamp(0.0, 1.0)        
                
                diff_color = gb.color * gb.diffuse

                if lighting.ambient_color is not None:
                    ambient = lighting.ambient_color.reshape(3, 1, 1)
                    diff_color = diff_color + ambient

                gb.final_color = gb.final_color*(1.0 - gb.alpha) + diff_color*gb.alpha

        return gb
    