
'''
#this file is used to generate adversarial examples based on the optimized weight of the generator.
the file path of test images should be provided: test_dir = r'the file path of test images'
'''
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
import torch.nn as nn
from torch import Tensor
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class MultiplyByHalf(nn.Module):
    def forward(self, x):
        x=x * 0.5
        return x
class myTanh(nn.Tanh):
    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(0.2*input)#
        
   
#----------the generator--------------#
img_shape = (3,1938)
latent_dim=100
noise_BS=32
class Generator_V2(nn.Module):
    def __init__(self):
        super(Generator_V2, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 256, normalize=True),
            *block(256, 256, normalize=True),
            *block(256, 256, normalize=True),
        )
        self.Linear= nn.Linear(256, int(np.prod(img_shape)))
        self.tanh=myTanh()
        self.Mhalf=MultiplyByHalf()

    def forward(self, z):
        img = self.model(z)
        img = self.Linear(img)
        img = self.tanh(img)
        img = self.Mhalf(img)
        img = img.view(img.shape[0], *img_shape)
        return img
    
    
class MyDataset_Audi(Dataset):
    def __init__(self,mesh, data_dir, img_size,  distence=None, mask_dir='', ret_mask=False,device=''):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']
                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
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

    def get_params(self,carlaTcam, carlaTveh):  # carlaTcam: tuple of 2*3  'cam_trans':[[x,y,z],[pitch, yaw, roll]]
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
        scale=5.3
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
        self.renderer.shader.lights=DirectionalLights(device=self.device, direction=[direction])#[list(np.array(eye)*-1)]   cam_trans[0]  [list(camera_direction*-1)]

        materials = Materials(
                    device=self.device,
                    specular_color=[[1.0, 1.0, 1.0]],
                    shininess=500.0
                )
        
        self.renderer.rasterizer.cameras=train_cameras
        self.renderer.shader.cameras=train_cameras
        
        # self.set_cameras(train_cameras)
        images = self.renderer(self.mesh,materials=materials )      
        imgs_pred = images[:, ..., :3]#.permute(0, 3, 1, 2)#
        # save_image(imgs_pred[-1,:,:,:].unsqueeze(0).permute(0, 3, 1, 2).cpu().detach(), "TEST_RENDER_116.png")
        
        img = img[:, :, ::-1]
        img_cv = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0   


        # img_tensor_clean=img.clone()
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
        new_contour = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_car = torch.zeros(img.permute(0, 2,3, 1).shape, device=self.device)
        new_car[:,:,:,0] = F.pad(imgs_pred[:,:,:,0], location, "constant", value=0)
        new_car[:,:,:,1] = F.pad(imgs_pred[:,:,:,1], location, "constant", value=0)
        new_car[:,:,:,2] = F.pad(imgs_pred[:,:,:,2], location, "constant", value=0)#
        
        total_img = torch.where((new_contour == 0.),img.permute(0, 2,3, 1), new_car)

        '''-------------------get adversarial images to be evaluated-------------'''
        img_save_dir=r'Evaluation_imgs/'
        save_image(total_img[0,:,:,:].unsqueeze(0).permute(0, 3, 1, 2).cpu().detach(), img_save_dir + file_name+'.jpg')

        return index,file, total_img.squeeze(0) , imgs_pred.squeeze(0),new_contour.squeeze(0)
    
    def __len__(self):
        return len(self.files)

def initialize_patch(mesh,device,texture_atlas_size):
    print('Initializing patch...')
    sampled_planes = list()
    with open(r'BMW_butt_lagger_tri\BMW_QZH.txt', 'r') as f:
        face_ids = f.readlines() #
        for face_id in face_ids:
            continue  
            if face_id != '\n':
                sampled_planes.append(int(face_id))              
    idx = torch.Tensor(sampled_planes).long().to(device)
    patch = torch.rand(len(sampled_planes), texture_atlas_size,texture_atlas_size, 3, device=(device), requires_grad=True)#

    return patch,idx



if __name__ == '__main__':
    import random
    seed=1234 #1111#      
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    import tqdm
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    test_dir = r'E:\NetDisk2\DualAttentionAttack\phy_attack\\test_carlaTestZY/'
    aux_net_weight=r'auxNetWeight\auxNet_epoch_9.pt'##
    aux_net_weight_SWA=r'auxNetWeight\auxNet_epoch_9_SWA.pt'

    mesh_dir=r'Audi\Audi_rect'
    
    '''#-------------load the weight of the generator-----------------'''
    netG_V2 = Generator_V2().cuda()
    checkpoint=torch.load(aux_net_weight)
    if aux_net_weight_SWA!=None:
        checkpoint_SWA=torch.load(aux_net_weight_SWA)
        new_dict = {}
        for key in checkpoint_SWA.keys():
            if 'module.model' in key:
                new_key = key.replace('module.model', 'model')
                new_dict[new_key] = checkpoint_SWA[key]
            elif 'module.Linear' in key:
                new_key = key.replace('module.Linear', 'Linear')
                new_dict[new_key] = checkpoint_SWA[key]
            # else:
            #      new_dict[key] = checkpoint_SWA[key]
        checkpoint['model'] = new_dict
    netG_V2.load_state_dict(checkpoint)
  
    # Generate batch of latent vectors
    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (noise_BS, latent_dim)))
    _patch = netG_V2(noise).mean(dim=0,keepdim=True)
    _patch = _patch + 0.5 
    total_patch1=_patch.clone().squeeze().permute(1, 0).unsqueeze(1).unsqueeze(1)#[N, 1, 1, 3]
 

    texture_atlas_size=1
    mesh_dataset = MeshDataset(mesh_dir, device, texture_atlas_size=texture_atlas_size, max_num=1)
    for mesh in mesh_dataset:
        patch = total_patch1  
        idx =   torch.load(r'idx_save_0.pt').to(device) 
        
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
            pass
        
    