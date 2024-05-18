
from math import pi
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torchvision.utils import save_image
import imageio
from PIL import Image
from MeshDataset import MeshDataset
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader,
    materials
)
import math
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
# from myRandomAffine import myRandomAffine
from utils2 import myRandomSunFlare 
import albumentations as A #
import albumentations.augmentations.functional as AF
from albumentations.augmentations.functional import from_float,to_float
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyDataset_Audi(Dataset):
    def __init__(self,mesh, data_dir, img_size, device=''):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        self.device=device
        self.mesh=mesh
        raster_settings = RasterizationSettings(
            image_size= self.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            max_faces_per_bin=26000
        )

        lights = PointLights(device=self.device, location=[[100.0, 85, 100.0]])
        self.cameras=''
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=lights
            )
        )
    def set_cameras(self, cameras):
        self.cameras = cameras #
    def set_mesh(self, mesh):
        self.mesh = mesh 

    def get_params(self,carlaTcam, carlaTveh):   
        scale =0.4 
        eye = [0, 0, 0]
        for i in range(0, 3):
            eye[i] = carlaTcam[0][i] * scale 
        pitch = math.radians(carlaTcam[1][0]) 
        yaw = math.radians(carlaTcam[1][1])
        roll = math.radians(carlaTcam[1][2])
        cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        cam_up = [math.cos(math.pi/2+pitch) * math.cos(yaw), math.cos(math.pi/2+pitch) * math.sin(yaw), math.sin(math.pi/2+pitch)]

        p_cam = eye
        p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
        p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
        p_l = [p_cam, p_dir, p_up]
        trans_p = []
        for p in p_l:
            if math.sqrt(p[0]**2 + p[1]**2) == 0:
                cosfi = 0
                sinfi = 0
            else:
                cosfi = p[0] / math.sqrt(p[0]**2 + p[1]**2)
                sinfi = p[1] / math.sqrt(p[0]**2 + p[1]**2)        
            cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
            sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
            trans_p.append([math.sqrt(p[0]**2 + p[1]**2) * cossum, math.sqrt(p[0]**2 + p[1]**2) * sinsum, p[2]])
        return trans_p[0], \
            [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
            [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]
   
    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img'] 
        veh_trans = data['veh_trans'] #
        cam_trans = data['cam_trans']
        file_name=file.split('/')[-1].split('.npz')[0]   
        scale=4.2
        
        for i in range(0, 3):
            cam_trans[0][i] = cam_trans[0][i] * scale 
        eye, camera_direction, camera_up = self.get_params(cam_trans, veh_trans)
        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 0, 0),))
        R[:,:,0]=R[:,:,0]*-1  
        R[:,0,:]=R[:,0,:]*-1
        tmp=R[:,1,:].clone()
        R[:,1,:]=R[:,2,:].clone()
        R[:,2,:]=tmp

        tmp1=R[:,0,:].clone()
        R[:,0,:]=R[:,2,:].clone()
        R[:,2,:]=tmp1
        R[:,0,:]=R[:,0,:]*-1

        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,znear=1.0,zfar=300.0,fov=45.0)
        direction=list(1*np.array(torch.bmm(R,torch.from_numpy(np.array(camera_direction)).unsqueeze(0).unsqueeze(2).float()).squeeze()))
        self.renderer.shader.lights=DirectionalLights(device=self.device, direction=[direction])
        materials = Materials(
                    device=self.device,
                    specular_color=[[1.0, 1.0, 1.0]],
                    shininess=100#500.0
                )
        
        self.renderer.rasterizer.cameras=train_cameras
        self.renderer.shader.cameras=train_cameras

        images = self.renderer(self.mesh,materials=materials )      
        imgs_pred = images[:, ..., :3]
      
        img = img[:, :, ::-1]
        img_cv = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0).float()
        img /= 255.0  

        bg_shape = img.shape
        car_size=self.renderer.rasterizer.raster_settings.image_size
        dH = bg_shape[2] - car_size
        dW = bg_shape[3] - car_size
        location = (
            dW // 2, #+ x_translation,
            dW - (dW // 2), #- x_translation,
            dH // 2, #+ y_translation,
            dH - (dH // 2) #- y_translation
        )#(0, 0, 0, 0)
        contour = torch.where((imgs_pred == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        

        
        new_contour = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)#new_contour.shape:torch.Size([10, 416, 416, 3])
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_car = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)
        new_car[:,:,:,0] = F.pad(imgs_pred[:,:,:,0], location, "constant", value=0)
        new_car[:,:,:,1] = F.pad(imgs_pred[:,:,:,1], location, "constant", value=0)
        new_car[:,:,:,2] = F.pad(imgs_pred[:,:,:,2], location, "constant", value=0)
 
        total_img = torch.where((new_contour == 0.),img.permute(0, 2,3, 1), new_car)
        
        '''#-----------------------light spots----------------------------------'''
        seed_everything(1234+index)
        pert_path=os.path.join('PatchOnlyImg_Audi',file.split('\\')[-1].split('.npz')[0]+'.jpg')
        pert_img = cv2.imdecode(np.fromfile(pert_path, dtype=np.uint8), cv2.IMREAD_COLOR)#
        indices = np.where((pert_img[:,:,0]+ pert_img[:,:,1]+ pert_img[:,:,2])>120 )
        coordinates = zip(indices[1], indices[0])
        unique_coordinates = list(set(list(coordinates)))
        for ii in range(len(unique_coordinates)):
            cv2.drawMarker(pert_img,position=unique_coordinates[ii],color=(0, 0, 255),markerSize = 1, markerType=cv2.MARKER_CROSS, thickness=1)
        rect=cv2.minAreaRect(np.array(unique_coordinates)) 
        # box = np.int0(cv2.boxPoints(rect)) 
        src_radius=int(min(rect[1])*0.6)
        flare_roi=random.choice(unique_coordinates) 
        flare_roi_x,flare_roi_y=flare_roi[0]/608.,flare_roi[1]/608.
        t=myRandomSunFlare( 
                flare_roi=(flare_roi_x-0.001, flare_roi_y-0.001, flare_roi_x+0.001, flare_roi_y+0.001),#flare_roi: (x_min, y_min, x_max, y_max)
                # flare_roi=(0, 0.75, 1, 1),
                # flare_roi=(0.4, 0.4, 0.6, 0.6),
                src_radius=src_radius,
                always_apply=False,
                p=0.5)#AL.add_sun_flare
        img_black = np.zeros((608, 608, 3), dtype = np.uint8)
        img_flare = t(image=img_black)["image"]
        img_flare = torch.from_numpy(img_flare).to("cuda").div(255.0).unsqueeze(0)
        img_flare = img_flare#.permute(0, 3, 1, 2)

        total_img = total_img + img_flare
        total_img = torch.clamp(total_img,0,1)
        #---------------------------------------------------------------------------------#
        
        

        '''#-------------------------------shadows---------------------------------#'''
        seed_everything(1234+index)
        assert total_img.shape[0]==1, 'wrong!'
        img_tmp=total_img.clone().detach().cpu().squeeze()
        img_total_np = img_tmp.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
        # t_shadow=A.RandomShadow(num_shadows_upper=1,shadow_dimension=4)
        # t_shadow=A.RandomShadow(num_shadows_upper=1,shadow_dimension=4,shadow_roi=(0, 0.75, 1, 1),p=0.5)
        t_shadow=A.RandomShadow(num_shadows_upper=1,shadow_dimension=4,shadow_roi=(0.4, 0.4, 0.6, 0.6),p=0.5)
        img_shadow=t_shadow(image=img_total_np)["image"]
        img_shadow = torch.from_numpy(img_shadow).to("cuda").div(255.0).unsqueeze(0)#shape=[1, 608, 608, 3]
        img_shadow = img_shadow#.permute(0, 3, 1, 2)
        # save_image(img_shadow[0,:,:,:].unsqueeze(0).permute(0, 3, 1, 2).cpu().detach(), 'test_1234.png')
        total_img = total_img-total_img.data + img_shadow
        total_img = torch.clamp(total_img,0,1)
        # -------------------------------------------------------------------------------------------#        
        
      

        '''#----------------------------------raindrops---------------------------------------------#
        seed_everything(1234+index)
        assert total_img.shape[0]==1, 'wrong!'
        img_tmp=total_img.clone().detach().cpu().squeeze()#
        img_total_np = img_tmp.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()#
        t_Spatter=A.Spatter(cutout_threshold = 0.7) 
        # t_Spatter=A.Spatter(cutout_threshold = 0.68,mode='mud')  
        img_Spatter=t_Spatter(image=img_total_np)["image"]
        img_Spatter = torch.from_numpy(img_Spatter).to("cuda").div(255.0).unsqueeze(0)#shape=[1, 608, 608, 3]
        img_Spatter = img_Spatter#.permute(0, 3, 1, 2)
        # save_image(img_Spatter[0,:,:,:].unsqueeze(0).permute(0, 3, 1, 2).cpu().detach(), 'test_1234.png')
        total_img = total_img-total_img.data + img_Spatter#
        total_img = torch.clamp(total_img,0,1)
        # -------------------------------------------------------------------------------------------#        
        '''

        return index,file, total_img.squeeze(0) , imgs_pred.squeeze(0),new_contour.squeeze(0)
    
    def __len__(self):
        return len(self.files)


def initialize_patch(mesh,device,texture_atlas_size):
    print('Initializing patch...')
    sampled_planes = list()  

    with open(r'XXX.txt', 'r') as f:
        face_ids = f.readlines() 
        for face_id in face_ids:
            continue  
            if face_id != '\n':
                sampled_planes.append(int(face_id))              
    idx = torch.Tensor(sampled_planes).long().to(device)
    patch = torch.rand(len(sampled_planes), texture_atlas_size,texture_atlas_size, 3, device=(device), requires_grad=True)#
    return patch,idx


class myRandomShadow(A.RandomShadow):
    def apply(self, image, vertices_list=(), **params):
        return myadd_shadow(image, vertices_list)#AF.add_shadow(image, vertices_list)



def myadd_shadow(img, vertices_list):
    """Add shadows to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        vertices_list (list):

    Returns:
        numpy.ndarray:

    """
    # non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomShadow augmentation".format(input_dtype))

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(img)

    # adding all shadow polygons on empty mask, single 255 denotes only red channel
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)

    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    red_max_value_ind = mask[:, :, 0] == 255
    # image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.5 #初始
    # image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.25
    image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb

if __name__ == '__main__':
    import tqdm
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    train_dir = r''
    test_dir = r''
  
    mesh_dir=r'Audi'
    texture_atlas_size=1
    mesh_dataset = MeshDataset(mesh_dir, device, texture_atlas_size=texture_atlas_size, max_num=1)
    for mesh in mesh_dataset:
        patch,idx=initialize_patch(mesh,device,texture_atlas_size)#''''''
        texture_image=mesh.textures.atlas_padded()
        clamped_patch = patch.clone().clamp(min=1e-6, max=0.99999)
        mesh.textures._atlas_padded[:,idx,:,:,:] = clamped_patch  
        mesh.textures.atlas = mesh.textures._atlas_padded
        mesh.textures._atlas_list = None
     
        dataset = MyDataset_Audi(mesh,test_dir, 608, device=device)
        loader = DataLoader(
            dataset=dataset,     
            batch_size=1,
            shuffle=False,        
            ) 
        tqdm_loader = tqdm.tqdm(loader)
        for i, (index,file_name, total_img, texture_img,contour) in enumerate(tqdm_loader):
            # print(total_img.size(),texture_img.size())
            pass
        
    