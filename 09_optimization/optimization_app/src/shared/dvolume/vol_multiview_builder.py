#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#for use only in the context of the "Machine Aesthetics" course at GSD. Do not share or distribute
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com

#pip install vtk

from typing import Callable, Optional, Union
import torch
import numpy as np
import cv2
import random

from .vol import *
from .vol_types import *
from .vol_field import *
from .vol_coords import *
from .vol_camera import *
from .vol_bounds import *
from .vol_transform3d import *
from .vol_lighting import *
from .vol_viewport import *
from .vol_model import *
import numbers
#__________________________________________________________________
class OptimizerType(Enum):
    Adam = 0
    SGD = 1
    AdamW = 2


UpdateableWithRange = Union[
    list[float] , 
    tuple[float, float] ,  #min, max -> random
    tuple[float, float, float] #min, max, step
]

AnyUpdateable = Union[float, 'Updateable', list]
AnyUpdateableDef = Union[UpdateableWithRange , AnyUpdateable]

CallableLoss = Callable[[torch.Tensor], torch.Tensor]

class Updateable: 
    @staticmethod
    def make(value : AnyUpdateableDef, update_every : int = 1) -> AnyUpdateable:
        if value is None:
            return None
        
        if isinstance(value, Updateable) or isinstance(value, numbers.Number):
            return value

        return Updateable(value, update_every)
    
    @staticmethod
    def can_update(value : AnyUpdateable) -> bool:
        if value is None:
            return False
        
        if isinstance(value, list):
            return any(Updateable.can_update(v) for v in value)
        
        return hasattr(value, 'update_for_epoch')
    
    @staticmethod
    def ensure_update(value : AnyUpdateable, epoch : int) -> bool:      
        if not Updateable.can_update(value):
            return False
        
        if isinstance(value, list) or isinstance(value, tuple):
            changed = False
            for v in value:
                changed = Updateable.ensure_update(v, epoch) or changed

            return changed
        
        return value.update_for_epoch(epoch)
    
    @staticmethod
    def extract_values(value: AnyUpdateable) -> Union[float, list[float]]:
        if value is None:
            return 0.0
        
        if isinstance(value, list) or isinstance(value, tuple):
            return [Updateable.extract_values(v) for v in value]
        
        if isinstance(value, numbers.Number):
            return value
        
        if isinstance(value, Updateable):
            return value.value
        
        raise ValueError('Invalid value for Updateable')
    
    def __init__(self, value : UpdateableWithRange, update_every : int = 1):
        self.update_every : int = update_every
        self.range : tuple[float, float] = (0.0, 1.0)
        self.step : float = 0.0
        self.value : float = 0.0

        self.last_epoch : int = -1

        if len(value) == 2:
            self.range = (value[0], value[1])
            self.value = random.uniform(value[0], value[1])
        elif len(value) == 3:
            self.range = (value[0], value[1])
            self.step = value[2]
            self.value = value[0]
        else:
            raise ValueError('Invalid value for Updateable')
    
    @property
    def isRandom(self) -> bool:
        return self.step == 0.0
    
    def update_for_epoch(self, epoch : int) -> bool:      
        if self.last_epoch == epoch or epoch % self.update_every != 0:
            return False

        if self.isRandom:
            self.value = random.uniform(self.range[0], self.range[1])
        else:
            self.value += self.step
            if self.value > self.range[1]:
                self.value = self.range[0] + (self.value - self.range[1])

        self.last_epoch = epoch
        return True



class View:
    def __init__(self, 
                    width : int, height : int, depth_layers : int,
                    background : Union[Vec3, torch.Tensor], #{3} | {3, H, W}
                    camera : VolumeCamera,
                    device : torch.device,    
                    *,
                    fg_color : Optional[torch.Tensor] = None,           
                    depth_noise : float = VolumeViewport.DEFAULT_NOISE,                    
                    lighting : Optional[VolLighting] = None,
                    compositing_mode : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                    volume_opacity : float = VolumeViewport.DEFAULT_OPACITY,
                    volume_density_contrast : float = VolumeViewport.DEFAULT_SIGMOID_FACTOR,
                    fog_color : Optional[Vec3] = None,
                    weight : float = 1.0,
                    require_alpha : bool = False,
                    require_position : bool = False,
                    require_normals : bool = False,
                    use_GBuffers : bool = False
                 ):
        self.device : torch.device = device

        self.background : torch.Tensor = ensureTensor(background, self.device)
        self.fg_color : Optional[torch.Tensor] = fg_color
        self.camera : VolumeCamera = camera        
        self.losses : list[CallableLoss] = []
        self.lighting : Optional[VolLighting] =  lighting
        self.compositing_mode : VolumeRendererCompositing = compositing_mode

        self.title = ''    

        self.volume_opacity = volume_opacity
        self.volume_density_contrast = volume_density_contrast

        self.fog_color : Optional[Vec3] = fog_color

        self.require_alpha = require_alpha
        self.require_position = require_position
        self.require_normals = require_normals
        
        self.use_GBuffers = use_GBuffers or (lighting is not None and lighting.isPointLight) or self.require_position or self.require_normals

        self.weight : float = weight

        self.viewport : VolumeViewport = VolumeViewport(
            camera, width, height, depth_layers, depth_noise
        )

        self.GBuffers : GBuffers = None
        self.last_canvas : torch.Tensor = None
        self.last_loss : float = 0.0

        self.loss_history : list[float] = []

    def getOpenCVImage(self, plot_history: bool = False, min_max: tuple[float, float] = (-50.0, 50.0)) -> np.ndarray:
        if self.last_canvas is None:
            return None
        
        img = self.last_canvas.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0.0, 1.0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if plot_history and len(self.loss_history) > 1:
            cv2.putText(img, f'{self.last_loss:.3f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            w = img.shape[1]
            h = img.shape[0]
            dx = w/len(self.loss_history)
            x0 = 0.0
            x1 = 0.0
            yscale = h/(min_max[1] - min_max[0])
            for i in range(1, len(self.loss_history)):
                x1 += dx
                l = self.loss_history[i]
                y0 = h - (self.loss_history[i-1] - min_max[0]) * yscale
                y1 = h - (l - min_max[0]) * yscale 
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1),), (255, 255, 255), 1, cv2.LINE_AA)
                x0 = x1





        if self.title != '':
            cv2.putText(img, self.title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img
    
    def update_for_epoch(self, epoch) -> bool:
        return Updateable.ensure_update([self.lighting, self.camera], epoch) 
    
    def reset(self):
        self.last_loss = 0.0
        self.loss_history = []
    
    def render(self, volume : VolumeModel) -> torch.Tensor:
        if self.use_GBuffers:
            self.GBuffers = self.viewport.renderVolumeUsingGBuffers(
                volume,
                background= self.background,
                volume_color = self.fg_color,
                compositing_opacity = self.volume_opacity,
                density_sigmoid_factor = self.volume_density_contrast,
                lighting = self.lighting,
                recompute_depth_bounds=False,
                volume_compositing=self.compositing_mode,
                alpha_compositing=self.compositing_mode,
                require_position=self.require_position,
                require_normals=self.require_normals,
                fog_color=self.fog_color,
            )                
        else:
            self.GBuffers = self.viewport.renderVolume(
                volume, 
                background= self.background, 
                volume_color = self.fg_color,
                compositing_opacity = self.volume_opacity,
                density_sigmoid_factor = self.volume_density_contrast,
                lighting = self.lighting,
                recompute_depth_bounds=False,
                volume_compositing=self.compositing_mode,
                alpha_compositing=self.compositing_mode,
                require_alpha=self.require_alpha,
                fog_color=self.fog_color
            )
    
        self.last_canvas = self.GBuffers.final_color
        return self.last_canvas


class MultiviewBuilder:
    def __init__(self, volume_resolution : Idx3, device: torch.device, viewport_res : tuple[int, int] = (256, 256)):
        
        self.device : torch.device = device

        Vol.device = self.device

        self.volume_resolution : Idx3 = volume_resolution
        self.viewport_width : int = viewport_res[0]
        self.viewport_height : int =  viewport_res[1]

        self.volume : VolumeModel = VolumeModel(self.volume_resolution, device=self.device)
       
        self.views : list[View] = []

        self.voxel_optimizer : torch.optim.Optimizer = None
        self.param_optimizer : torch.optim.Optimizer = None

        self.optimization_voxel_targets : list[torch.Tensor] = []
        self.optimization_param_targets : list[torch.Tensor] = []

        self.epoch :int = 0
        self.loss_history : list[float] = []
        
    #lighting
    def createLighting(self, 
                            dir_x : AnyUpdateableDef, 
                            dir_y : AnyUpdateableDef, 
                            dir_z : AnyUpdateableDef,
                            isPointLight : bool = False,
                            update_every : int = 1,
                            ambient_color : Optional[Vec3] = None,
                            specular_power : float = 16.0    
                       ) -> VolLighting:
        
        dir_x_ = Updateable.make(dir_x, update_every)
        dir_y_ = Updateable.make(dir_y, update_every)
        dir_z_ = Updateable.make(dir_z, update_every)

        updateables = [dir_x_, dir_y_, dir_z_]
        values = Updateable.extract_values(updateables)

        ambient_color = ensureTensor(ambient_color, self.device) if ambient_color is not None else None
        light_dir = ensureTensor(values, self.device)
        lighting = VolLighting(
                        light_dir, 
                        ambient_color = ambient_color, 
                        isPointLight=isPointLight, 
                        specular_power=specular_power, 
                        device=self.device
                    ) 

        if Updateable.can_update(updateables):
            def update_for_epoch(epoch):
                changed = Updateable.ensure_update(updateables, epoch)
                if changed:
                    with torch.no_grad():
                        values = Updateable.extract_values(updateables)
                        lighting.setDirection(values[0], values[1], values[2])
                
                return changed

            lighting.update_for_epoch = update_for_epoch

        return lighting
        

    #cameras
    def createCameraWithAngles(self,                         
                        angle_xy_deg: AnyUpdateableDef,
                        angle_z_deg: AnyUpdateableDef,
                        distance: AnyUpdateableDef,
                        target_point: list[AnyUpdateableDef] = [0.0, 0.0, 0.0],
                        up_vector: list[AnyUpdateableDef] = [0.0, 0.0, 1.0],
                        fov_deg: AnyUpdateableDef = 50.0,
                        aspect = 1.0,
                        znear = 0.01,
                        zfar = 10.0,
                        update_every: int = 1,
                        orthographic : bool = False,
                        ortho_half_width : AnyUpdateableDef = 3.0                  
                     ) -> VolumeCamera:
        
        
        distance_ = Updateable.make(distance, update_every)
        angle_xy_deg_ = Updateable.make(angle_xy_deg, update_every)
        angle_z_deg_ = Updateable.make(angle_z_deg, update_every)
        target_point_ = [Updateable.make(v, update_every) for v in target_point]
        up_vector_ = [Updateable.make(v, update_every) for v in up_vector]
        fov_deg_ = Updateable.make(fov_deg, update_every)
        ortho_half_width_ = Updateable.make(ortho_half_width, update_every)

        updateables = [distance_, angle_xy_deg_, angle_z_deg_, target_point_, up_vector_, fov_deg_, ortho_half_width_]
        values = Updateable.extract_values(updateables)


        camera = VolumeCamera.makeWithAngles(
            values[0], values[1], values[2],
            values[3], values[4], values[5], aspect, znear, zfar, self.device, orthographic, values[6])
        
        if Updateable.can_update(updateables):
            def update_for_epoch(epoch):
                changed = Updateable.ensure_update(updateables, epoch)
                if changed:
                    with torch.no_grad():
                        values = Updateable.extract_values(updateables)
                        camera.setWithAngles(
                            values[0], values[1], values[2],
                            values[3], values[4], values[5], aspect, znear, zfar, orthographic, values[6])
                        camera.update()
                        
                return changed
                        
            camera.update_for_epoch = update_for_epoch

        return camera
    
    #background -> {3, viewport_height, viewport_width}
    def createBackgroundImage(self, color: Optional[Vec3] = None) -> torch.Tensor:
        if color is None:
            return torch.zeros(3, self.viewport_height, self.viewport_width, device=self.device)
        else :
            rgb = ensureTensor(color, self.device)
            return rgb.unsqueeze(1).unsqueeze(2).expand(3, self.viewport_height, self.viewport_width)

    def createBackgroundImageFromFile(self, image_file: str) -> torch.Tensor:
        cv2_img = cv2.imread(image_file)
        cv2_img = cv2.resize(cv2_img, (self.viewport_width, self.viewport_height))
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        bg_image = torch.tensor(cv2_img, dtype=torch.float32).permute(2,0,1) / 255.0
        return bg_image.to(self.device)
  
    #views
    def addViews(self, 
                    cameras: list[VolumeCamera], 
                    background : Union[Vec3 , torch.Tensor], #{3} | {3, H, W}                  
                    *,
                    fg_color : Optional[torch.Tensor] = None,       
                    depth_layers : int = VolumeViewport.DEFAULT_DEPTH_LAYERS,    
                    depth_noise : float = VolumeViewport.DEFAULT_NOISE,                    
                    lighting : Optional[VolLighting] = None,
                    compositing_mode : VolumeRendererCompositing = VolumeRendererCompositing.ALPHA,
                    volume_opacity : float = VolumeViewport.DEFAULT_OPACITY,
                    volume_density_contrast : float = VolumeViewport.DEFAULT_SIGMOID_FACTOR,
                    weight : float = 1.0,
                    require_alpha : bool = False,
                    use_GBuffers : bool = False,
                    losses : Union[Optional[CallableLoss], Optional[list[CallableLoss]]] = None,
                    fog_color : Optional[Vec3] = None
                 ) -> list[View]:
        
        new_views = []
        
        for camera in cameras:
            view = View(
                self.viewport_width, self.viewport_height, depth_layers, 
                background,
                camera, self.device,
                fg_color = fg_color,                
                depth_noise = depth_noise,
                lighting = lighting,
                compositing_mode = compositing_mode,
                volume_opacity = volume_opacity,
                volume_density_contrast = volume_density_contrast,
                weight = weight,
                require_alpha = require_alpha,
                use_GBuffers=use_GBuffers,
                fog_color = fog_color
                )
            
            if losses is not None:
                if isinstance(losses, list):
                    view.losses.extend(losses)
                else:
                    view.losses.append(losses)
            
            new_views.append(view)
            self.views.append(view)

        return new_views

    #fields
    def setDensityField(self, density_field: Union[torch.Tensor , VolumeField]):
        if isinstance(density_field, VolumeField):
            self.volume.density = density_field
        elif isinstance(density_field, torch.Tensor):
            self.volume.density = VolumeField(density_field)
        else:
            raise ValueError('Invalid density field type')
        
        self.volume.invalidateDensity()
        self.volume.invalidateNormals()
        self.volume.invalidateGradient()
        self.volume.resolution = (self.volume.density.nx, self.volume.density.ny, self.volume.density.nz)

    def setColorField(self, color_field: Union[torch.Tensor , VolumeField]):
        if isinstance(color_field, VolumeField):
            self.volume.color = color_field
        elif isinstance(color_field, torch.Tensor):
            self.volume.color = VolumeField(color_field)
        else:
            raise ValueError('Invalid color field type')

    #optimizer configuartion
    def registerOptimizableVolumeTensor(self, tensor: Union[torch.Tensor , VolumeField]) -> torch.Tensor:
        if isinstance(tensor, VolumeField):
            tensor = tensor.tensor

        if not any(tensor is t for t in self.optimization_voxel_targets): 
            self.optimization_voxel_targets.append(tensor)

        return tensor


    def registerOptimizableParamsTensor(self, tensor: Union[torch.Tensor , list[float]]) -> torch.Tensor:
        tensor = ensureTensor(tensor, self.device)

        if not any(tensor is t for t in self.optimization_param_targets):
            self.optimization_param_targets.append(tensor)

        return tensor
    
    def __createOptimizer(self, params: list[torch.Tensor], learning_rate: float, momentum: float = 0.9, optimizer_type: OptimizerType = OptimizerType.Adam) -> torch.optim.Optimizer:
        if optimizer_type == OptimizerType.Adam:
            return torch.optim.Adam(params, lr=learning_rate)
        elif optimizer_type == OptimizerType.SGD:
            return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=True)
        elif optimizer_type == OptimizerType.AdamW:
            return torch.optim.AdamW(params, lr=learning_rate)
        else: #callable
            return optimizer_type(params, learning_rate)
        
    def __initializeTargets(self, targets: list[torch.Tensor]):
        with torch.no_grad():
            for t in targets:
                t.requires_grad = True
                t.initial_value = t.clone()

    def __resetTargets(self, targets: list[torch.Tensor]):
        with torch.no_grad():
            for t in targets:
                t.copy_(t.initial_value)

    def initializeVoxelOptimizer(self, learning_rate: float, momentum: float = 0.9, optimizer_type: OptimizerType = OptimizerType.Adam):
        if len(self.optimization_voxel_targets) == 0:
            return
        
        self.__initializeTargets(self.optimization_voxel_targets)
        self.voxel_optimizer = self.__createOptimizer(self.optimization_voxel_targets, learning_rate, momentum, optimizer_type)


    def initializeParamOptimizer(self, learning_rate: float, momentum: float = 0.9, optimizer_type: OptimizerType = OptimizerType.Adam):
        if len(self.optimization_param_targets) == 0:
            return
        
        self.__initializeTargets(self.optimization_param_targets)
        self.param_optimizer = self.__createOptimizer(self.optimization_param_targets, learning_rate, momentum, optimizer_type)


    #get clones of the current parameters
    def getParamsSnapshot(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        voxel_params = [t.clone() for t in self.optimization_voxel_targets]
        param_params = [t.clone() for t in self.optimization_param_targets]

        return voxel_params, param_params
    
    #set the parameters to the given values
    def setParamsSnapshot(self, vox_par: tuple[list[torch.Tensor], list[torch.Tensor]]):
        voxel_params, param_params = vox_par
        with torch.no_grad():
            if voxel_params is not None:
                for t, v in zip(self.optimization_voxel_targets, voxel_params):
                    t.copy_(v)

            if param_params is not None:
                for t, v in zip(self.optimization_param_targets, param_params):
                    t.copy_(v)
    
    #optimization steps
    def reset(self):
        self.epoch = 0
        self.__resetTargets(self.optimization_voxel_targets)
        self.__resetTargets(self.optimization_param_targets)
        self.loss_history = []
        for view in self.views:
            view.reset()

    def zero_grad(self):
        if self.voxel_optimizer is not None:
            self.voxel_optimizer.zero_grad()

        if self.param_optimizer is not None:
            self.param_optimizer.zero_grad()

        self.volume.invalidateDensity()

    def renderViews(self):
        for view in self.views:
            view.update_for_epoch(self.epoch)
            view.render(self.volume)

    def renderView(self, view_index: int) -> torch.Tensor:
        view = self.views[view_index]
        view.update_for_epoch(self.epoch)
        return view.render(self.volume)

    def computeLoss(self) -> torch.Tensor:
        loss = 0.0
        for view in self.views:
            view_loss = 0.0
            for l in view.losses:
                view_loss = l(view.last_canvas.unsqueeze(0)) + view_loss
            
            if isinstance(view_loss, torch.Tensor):
                view.last_loss = view_loss.item()
            else:
                view.last_loss = view_loss            

            view.loss_history.append(view.last_loss)

            loss = loss + view_loss * view.weight

        if isinstance(loss, torch.Tensor):
            self.loss_history.append(loss.item())
        else:
            self.loss_history.append(loss)  
        return loss

    def step(self, loss_eval: Optional[Callable[[], torch.Tensor]] = None):
        if self.voxel_optimizer is not None:
            self.voxel_optimizer.step(loss_eval)

        if self.param_optimizer is not None:
            self.param_optimizer.step(loss_eval)

        self.epoch += 1

    def getViewportsConcatenatedCVImage(self, plot_history: bool = False) -> np.ndarray:
        with torch.no_grad():
            images = []
            for view in self.views:
                img = view.getOpenCVImage(plot_history)
                if img is not None:
                    images.append(img)

        viewports_img = np.concatenate(images, axis=1)
        return viewports_img
    
    def showViewports(self, title: str, plot_history: bool = False):
        with torch.no_grad():
            viewports_img = self.getViewportsConcatenatedCVImage(plot_history)
            cv2.imshow('image', viewports_img)
            cv2.setWindowTitle('image', f'{title} {self.epoch}')
            cv2.waitKey(1)

    def saveSnapshot(self, folder_path: str):
         with torch.no_grad():
            
            self.volume.save(folder_path, 'volume', contours=[0.5])

            viewports_img = self.getViewportsConcatenatedCVImage()            
            image_file = folder_path + '.png'
            viewports_img = (viewports_img*255.0).astype(np.uint8)

            cv2.imwrite(image_file, viewports_img)

    
    