import os
import json
import torch

from .vol import *
from .vol_types import *
from .vol_field import *
from .vol_coords import *
from .vol_camera import *
from .vol_bounds import *
from .vol_transform3d import *
from .vol_lighting import *


class VolumeModel:
    def __init__(self, 
                 resolution : Idx3, #this is in xyz order
                 model_matrix : Transform3d = None,
                 device : torch.device = Vol.device
                 ):
        
        self.resolution = resolution
        self.setModelMatrix(model_matrix)

        self.density = VolumeField.makeSolidDensity(0.0, resolution, device)
        self.color : VolumeField = None
        self.__gradient : VolumeField = None     
        self.__normals : VolumeField = None
        self.__uv_coords = VolumeCoords(resolution, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], device=device)

        self.__gradient_needs_update = True
        self.__normals_needs_update = True

    def save(self, folder : str, name : str, contours : list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):

        if not os.path.exists(folder):
            os.makedirs(folder)

        info = {
            'resolution' : ensureList(self.resolution)
        }

        if self.model_matrix is not None:
            info['model_matrix'] = ensureList(self.model_matrix.tensor)

        with open(os.path.join(folder, name + '.json'), 'w') as f:
            json.dump(info, f)

        self.density.save(os.path.join(folder, name + '_density.npy'))
        if self.color is not None:
            self.color.save(os.path.join(folder, name + '_color.npy'))

        #create vtk field from density tensor then save mesh
        vtk_density = self.density.tensor.squeeze(0).squeeze(0).permute(2, 1, 0).cpu().numpy()
        vtk_density = vtk_np.numpy_to_vtk(vtk_density.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_density.SetName('Scalar')

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(self.resolution[2], self.resolution[1], self.resolution[0])
        vtk_image.GetPointData().AddArray(vtk_density)
        vtk_image.GetPointData().SetActiveScalars('Scalar')

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(os.path.join(folder, name + '_density.vti'))
        writer.SetInputData(vtk_image)
        writer.Write()

        #use vtkMarchingCubes to create a mesh from the density field
        mc = vtk.vtkImageMarchingCubes()
        mc.SetInputData(vtk_image)

        for contour in contours:
            mc.SetValue(0, contour)
            mc.Update()

            writer = vtk.vtkOBJWriter()
            writer.SetFileName(os.path.join(folder, name + '_density_' + str(contour) + '.obj'))
            writer.SetInputData(mc.GetOutput())
            writer.Write()




    @property
    def hasColor(self) -> bool:
        return self.color is not None
    
    @property
    def uv_coords(self) -> VolumeCoords:
        return self.__uv_coords
    
    @property
    def gradient_needs_update(self) -> bool:
        return self.__gradient_needs_update
    
    @property
    def normals_needs_update(self) -> bool:
        return self.__normals_needs_update
    
    def invalidateDensity(self):
        self.invalidateGradient()

    def invalidateGradient(self):
        self.__gradient_needs_update = True
        self.invalidateNormals()

    def invalidateNormals(self):
        self.__normals_needs_update = True
        
    @property
    def device(self):
        return self.density.device
    
    def setModelMatrix(self, model_matrix : Transform3d):
        self.model_matrix = model_matrix
        self.model_matrix_inv = model_matrix.getInverse() if model_matrix is not None else None
        self.normal_matrix = self.model_matrix_inv.tensor3x3.t() if model_matrix is not None else None
        self.invalidateNormals()
    
    def ensureGradient(self) -> VolumeField:
        if self.__gradient_needs_update:
            self.updateGradient()
        return self.__gradient

    def ensureNormals(self) -> VolumeField:
        if self.__normals_needs_update:
            self.updateNormals()

        return self.__normals

    def updateGradient(self):
        self.__gradient = self.density.getGradient()
        self.__gradient_needs_update = False

    def updateNormals(self):
        grad = self.ensureGradient()
        if self.normal_matrix is not None:
            grad_tensor = grad.tensor
            grad_tensor = grad_tensor.permute(0, 2, 3, 4, 1)
            grad_tensor = grad_tensor.reshape(self.resolution[0], self.resolution[1], self.resolution[2], 3)
            grad_tensor = torch.matmul(grad_tensor, self.normal_matrix.t())
            grad_tensor = grad_tensor.permute(3, 0, 1, 2)
            self.__normals = VolumeField(grad_tensor)
        else:
            self.__normals = grad

        self.__normals_needs_update = False

    def getDiffuseField(self, lighting : VolLighting) -> VolumeField:
        normals = self.ensureNormals()

        light_dir = lighting.light_dir.reshape(1, 3, 1, 1, 1)

        diffuse_tensor = torch.sum(normals.tensor * light_dir, dim=1, keepdim=True)
        diffuse_tensor = torch.clamp(diffuse_tensor*0.5+0.5, 0.0, 1.0)
        return VolumeField(diffuse_tensor)
    
    def getLitColorField(self, lighting : VolLighting) -> VolumeField:
        if lighting.isPointLight:
            raise ValueError("Point lights not supported use GBuffers.")                    

        diffuse = self.getDiffuseField(lighting)
        
        final_color : torch.Tensor = None

        if self.color is not None:
            final_color = self.color.tensor * diffuse.tensor
        else:
            final_color = torch.repeat_interleave(diffuse.tensor, 3, dim=1)

        if lighting.ambient_color is not None:
            ambient = lighting.ambient_color.reshape(1, 3, 1, 1, 1)
            final_color = final_color + ambient

        return VolumeField(final_color)
        
    def getViewBounds(self, camera : VolumeCamera) -> Bounds3d:
        return camera.getViewProjectedBounds(Bounds3d.makeUnitSymmetricCube(), self.model_matrix)
