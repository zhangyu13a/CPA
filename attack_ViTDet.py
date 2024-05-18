import os
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from MeshDataset import MeshDataset
from torchvision.utils import save_image
from tqdm import tqdm

import time
import torch.nn.functional as F
import argparse
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

import torch.nn as nn

from data_loader_nr_RAudi import MyDataset_Audi

from utils2 import TotalVariation_3d,myRandomAffine,SWAModel,EMAModel,T_ADEMAModel
from get_attention import netAttention
import random
import numpy as np
from random import choice
from torchvision.transforms.functional import InterpolationMode
from detectron2.config import LazyConfig, instantiate
from torch import Tensor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(1234)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.01)
        nn.init.constant_(m.bias.data, 0)

class MultiplyByHalf(nn.Module):
    def forward(self, x):
        x=x * 0.5
        return x

class myTanh(nn.Tanh):
    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(0.2*input)#

    
#----------build the generator--------------#
img_shape = (3,1938)#1938:the number of tri facets
latent_dim=100#"dimensionality of the latent space"
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


class Patch():
    def __init__(self, config_model,args, device):
        
        self.aux_net_V2=True
        self.aux_net_V2_SWA=True
        self.epoch_start=6#start to use SWA or EMA

        if self.aux_net_V2:
            self.netG_V2 = Generator_V2().cuda()
            self.netG_V2.apply(weights_init)                    
            self.swa_model = SWAModel(self.netG_V2, cuda=True) 

        self.args = args
        self.device = device
        self.cfg_model=config_model

        self.args = args
        self.device = device
        self.cfg_model=config_model

        # Datasets
        self.mesh_dataset = MeshDataset(self.args.mesh_dir, device, texture_atlas_size=self.args.texture_atlas_size, max_num=self.args.num_meshes)
        # MViTV2 model:
        self.dnet =  instantiate(self.cfg_model.model)
        self.dnet.to(self.cfg_model.train.device)
        self.dnet.eval()

        DetectionCheckpointer(self.dnet).resume_or_load(
            self.cfg_model.train.init_checkpoint, resume=False)

        self.net_attention = netAttention(self.dnet,self.cfg_model)    
     
        # Initialize adversarial patch
        self.patch = None
        self.idx = None
        if self.args.patch_dir is not None: 
          self.patch = torch.load(self.args.patch_dir + 'patch_save_0.pt').to(self.device)
          self.patch.requires_grad=True   
          self.idx = torch.load(self.args.patch_dir + 'idx_save_0.pt').to(self.device)

        if self.patch is None or self.idx is None:
            self.initialize_patch(device=self.device,texture_atlas_size=self.args.texture_atlas_size,tri_index_path=self.args.tri_index_path)#

        if self.aux_net_V2:
            self.patch_auxnet=None
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
   
    def attack(self):
        mesh = self.mesh_dataset.meshes[0]
        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)
        if self.aux_net_V2:
            optimizerG =  torch.optim.Adam(self.netG_V2.parameters(), lr=0.01, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam([self.patch], lr = 0.01)
        n_epochs=self.args.epochs
        small_score_num_pre=small_score_num=0
        for epoch in range(n_epochs): 
            
            small_score_num_pre=small_score_num
            print('small_score_num_pre=',small_score_num_pre)            
            for mesh in self.mesh_dataset:
                total_patch=self.patch.clone()#
                if self.aux_net_V2:
                    mesh=mesh.clone().detach()  
                    seed_everything(1234+epoch)  
                    # Generate batch of latent vectors
                    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (noise_BS, latent_dim)))
                    _patch = self.netG_V2(noise).mean(dim=0,keepdim=True)
                    _noise = torch.randn(_patch.shape[1::]).cuda() * 0.1
                    _patch = _patch + 0.5 + _noise#[1, 3, 1938]
                    total_patch1=_patch.clone().squeeze().permute(1, 0).unsqueeze(1).unsqueeze(1)#[N, 1, 1, 3]
                    self.patch_auxnet=total_patch1.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = total_patch1.clone().clamp(min=1e-6, max=0.99999)
                else:
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = total_patch.clamp(min=1e-6, max=0.99999)
    

                mesh.textures.atlas = mesh.textures._atlas_padded##
                mesh.textures._atlas_list = None
                dataset = MyDataset_Audi(mesh,self.args.train_dir, self.args.img_size, device=self.device)
                loader = DataLoader(
                    dataset=dataset,     
                    batch_size=self.args.batch_size,  
                    shuffle=self.args.shuffle,  
                    drop_last=self.args.drop_last,      
                    num_workers=0,  
                    worker_init_fn=np.random.seed(12)              
                    ) 

                tqdm_loader = tqdm(loader)
                small_score_num=0
                for i, (index,file_name_BS, total_img, texture_img,contour) in enumerate(tqdm_loader):
                    total_img = total_img.permute(0, 3, 1, 2)#[N H W C]->[N C H W] shape:torch.Size([1, 3, 608, 608])    
                    contour=contour.permute(0, 3, 1, 2)

                    save_image(total_img[0,:,:,:].unsqueeze(0).cpu().detach(), 'test_TotalImg.png')#TotalImg.shape:torch.Size([1, 3, 608, 608])
                  
                    assert total_img.shape[0]== contour.shape[0],''
                    total_img_with_countour=torch.cat((total_img,contour),dim=0)#
                    Rotation_all=myRandomAffine(degrees=10,scale=(0.9,1.1))
                    total_img_with_countour,_,_=Rotation_all(total_img_with_countour)
                    total_img=total_img_with_countour[0:total_img.shape[0]]
                    contour=total_img_with_countour[total_img.shape[0]:]

                    inputs = {"image": total_img,'contour':contour,'mask_num_batches':0,'clean_img_path':file_name_BS}
                    self.net_attention._register_hook()                
                    scores,mean_ious=self.net_attention(inputs,retain_graph=True)
                    self.net_attention.remove_handlers()

                    small_score_num=small_score_num+torch.sum(scores.detach()<0.5).item()#
                    if len(scores)<self.args.batch_size:
                        undetected_imgs=self.args.batch_size-len(scores)
                        small_score_num=small_score_num+undetected_imgs
                    if len(scores)==0:
                        log_dir=''                    
                        with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                            tqdm_loader.set_description('Epoch %d/%d ,small_score_num %d  ' % \
                            (epoch,n_epochs,small_score_num))#
                        continue
               
                    if self.aux_net_V2:
                        tv_loss = total_variation(self.patch_auxnet)*50
                    else:
                        tv_loss = total_variation(self.patch)*50
                    scores_sum=torch.sum(torch.mean(scores.clone(),dim=-1))                      
                    loss=torch.sum(scores_sum)*5+tv_loss*0.1

                    if self.aux_net_V2:
                        optimizerG.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizerG.step()
                        #SWA
                        if self.aux_net_V2_SWA and epoch>=self.epoch_start:                                
                            self.swa_model.update(self.netG_V2)                           
                    else:
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                                
                    if self.aux_net_V2:
                        lr=optimizerG.param_groups[0]['lr']
                    else:
                        lr=optimizer.param_groups[0]['lr']

                    log_dir=''                    
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                        tqdm_loader.set_description('Epoch %d/%d ,Loss %.3f ,tv_loss %.3f,small_score_num %d ,lr %.5f ' % \
                        (epoch,n_epochs,loss.data.cpu().numpy(),tv_loss.data.cpu().numpy(),small_score_num,lr))#bs改为2
                        if i==0:                            
                            f.write(time.strftime("%Y%m%d-%H%M%S")+'\n')
                        f.write('Epoch %d/%d ,Loss %.3f  ,tv_loss %.3f,small_score_num %d ,lr %.5f \n' % \
                        (epoch,n_epochs,loss.data.cpu().numpy(),tv_loss.data.cpu().numpy(),small_score_num,lr))
                    
                    texture_image = mesh.textures.atlas_padded()#
                    contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
                    brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
                    noise = torch.FloatTensor(self.patch.shape).uniform_(-1, 1) * self.noise_factor                  
                    augmented_patch = (self.patch.clone() * contrast) + brightness + noise.to(self.device)#([N, 1, 1, 3])
                    # Clamp patch to avoid PyTorch3D issues
                    clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)#[N, 1, 1, 3]
                   
                    if self.aux_net_V2:
                        mesh=mesh.clone().detach()  
                        seed_everything(1234+epoch+i+1)   
                        # Generate batch of latent vectors
                        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (noise_BS, latent_dim)))
                        # Generate fake image batch with G
                        # _patch = self.netG_V2(noise)
                        _patch = self.netG_V2(noise).mean(dim=0,keepdim=True)#在BS维度上求平均

                        _noise = torch.randn(_patch.shape[1::]).cuda() * 0.1
                        _patch = _patch + 0.5 + _noise#[1, 3, 1938]
                        
                        total_patch1=_patch.clone().squeeze().permute(1, 0).unsqueeze(1).unsqueeze(1)#[N, 1, 1, 3]
                        self.patch_auxnet=total_patch1.clone().clamp(min=1e-6, max=0.99999)
                        mesh.textures._atlas_padded[:,self.idx,:,:,:] = total_patch1.clone().clamp(min=1e-6, max=0.99999)
                  
                    else:
                        mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
                          
                    mesh.textures.atlas = mesh.textures._atlas_padded##   
                    mesh.textures._atlas_list = None
                    
                    dataset.set_mesh(mesh)    
  
            if small_score_num-small_score_num_pre<=0:  
                print('small_score_num-small_score_num_pre=',small_score_num-small_score_num_pre)
                if self.aux_net_V2:
                    optimizerG.param_groups[0]['lr']=optimizerG.param_groups[0]['lr']*0.5
                else:
                    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5

            if self.aux_net_V2:
                save_dir='auxNetWeight' 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path_patch=os.path.join(save_dir, 'auxNet_epoch_{}.pt'.format(epoch))
                torch.save(self.netG_V2.state_dict(), save_path_patch)
                if self.aux_net_V2_SWA and epoch>=self.epoch_start:                  
                    save_path_patch_SWA=os.path.join(save_dir, 'auxNet_epoch_{}_SWA.pt'.format(epoch))
                    torch.save(self.swa_model.state_dict(), save_path_patch_SWA)
            else:    
                # Save image and print performance statistics
                patch_save = self.patch.cpu().detach().clone()
                idx_save = self.idx.cpu().detach().clone()

                save_dir='patch_idx_save_dropout' 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path_patch=os.path.join(save_dir, 'patch_save_{}.pt'.format(epoch))
                save_path_idx=os.path.join(save_dir, 'idx_save_{}.pt'.format(epoch))
                torch.save(patch_save, save_path_patch)
                torch.save(idx_save, save_path_idx) 

     
    def initialize_patch(self,device,texture_atlas_size,tri_index_path):
        print('Initializing patch...')
        sampled_planes = list()  
        with open(tri_index_path, 'r') as f:
            face_ids = f.readlines()
            for face_id in face_ids:
                # continue  
                if face_id != '\n':
                    sampled_planes.append(int(face_id))              
        idx = torch.Tensor(sampled_planes).long().to(device)
        patch = torch.rand(len(sampled_planes), texture_atlas_size,texture_atlas_size, 3, device=(device), requires_grad=True)#
        self.idx = idx   
        self.patch = patch   
        self.patch_clone=patch.clone().detach()
        # return patch,idx        


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg = LazyConfig.load(args.config_file)#
    cfg = LazyConfig.apply_overrides(cfg, args.opts)# 
    cfg.model.roi_heads.box_predictor.test_nms_thresh=0.99
    cfg.model.roi_heads.box_predictor.test_score_thresh=0.05 
    cfg.model.roi_heads.box_predictor.test_topk_per_image=100 
    cfg.model.roi_heads.num_classes = 1
    cfg.train.init_checkpoint=args.weightfile#r'trained_weights\ViTDet_carla\model_final.pth'# 
    # cfg.freeze()
    return cfg

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--texture_atlas_size', type=int, default=1) 
    parser.add_argument('--tri_index_path', type=str, default=r'Audi\Audi_rect\Audi_rect.txt',help='')   
    parser.add_argument('--mesh_dir', type=str, default=r"Audi\Audi_rect")#
    parser.add_argument('--output', type=str, default='out/patch')
    parser.add_argument('--patch_dir', type=str, default=None,help='patch_dir is None normally, but it should be a path when resuming texture optimization from the last epoch')#default=''默认存放于根目录下
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=608)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--shuffle', type=bool, default=True,help='whether shuffle the data when training')#
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--num_meshes', type=int, default=1)
    parser.add_argument('--detector', type=str, default='Mvitv2')
    parser.add_argument('--printfile', type=str, default=r'non_printability\30values.txt')
    parser.add_argument('--train_dir', type=str, default=r'xxx')#
    parser.add_argument('--weightfile', type=str, default=r'trained_weights\ViTDet_carla\model_final.pth')#                                                 
    parser.add_argument(
        "--config-file",
        default=r'configs\COCO\mask_rcnn_vitdet_b_100ep.py',  
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0,
        help="Minimum score for instance predictions to be shown",
    )
 
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main():   
    seed_everything(1234)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg_model = setup_cfg(args)
    trainer = Patch(cfg_model,args, device)
    trainer.attack()

if __name__ == '__main__':
    main()