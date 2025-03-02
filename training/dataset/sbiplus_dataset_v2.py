import os
import pickle
import random
import sys
sys.path.append('.')
import torchvision
import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API
import albumentations as alb
from skimage.metrics import structural_similarity as ssim
import numpy as np
import logging
from dataset.utils.bi_online_generation import random_get_hull
from PIL import Image
c=0
def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))



def select_fake(fake, deepfake,bifake, fl, dfl,bil,bi_ratio=1, b_ratio=1, df_ratio=1,order='sequence'):
    fake_len = len(fake)
    df_len = len(deepfake)
    bi_len = bifake.size(0)
    # 这里为什么想着随机取？明明顺序也可以？难道说顺序的会好一些？？！？？顺序显然更方便对应和操作
    if order == 'random':
        fake_indexes = torch.randperm(len(fake))[:int(fake_len * b_ratio)]
        deepfake_indexes = torch.randperm(len(deepfake))[:int(df_len * df_ratio)]
        selected_fake = fake[fake_indexes]
        selected_deepfake = deepfake[deepfake_indexes]
        selected_fl = fl[fake_indexes]
        selected_dfl = dfl[deepfake_indexes]

    else:
        selected_fake = fake[:int(fake_len * b_ratio)]
        selected_deepfake = deepfake[:int(df_len * df_ratio)]
        selected_bifake = bifake[:int(bi_len * bi_ratio)]
        selected_fl = fl[:int(fake_len * b_ratio)]
        selected_dfl = dfl[:int(df_len * df_ratio)]
        selected_bil = bil[:int(bi_len * bi_ratio)]


    return torch.cat((selected_fake,selected_bifake, selected_deepfake), dim=0), torch.cat((selected_fl,selected_bil, selected_dfl), dim=0)

class SBIPlusV2Dataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        global c
        c=config
        self.config=config
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]
        self.fake_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label != 0]
        self.real_imgdict=self.format_imglist(self.real_imglist)
        self.fake_imgdict=self.format_imglist(self.fake_imglist)
        pop_item=self.fake_imgdict.pop('281')
        self.index_mapping=[]

        for each in self.fake_imgdict.keys():
            self.index_mapping.append(each)
        self.list_len = len(self.index_mapping)
        # 281 only exist for faceswap in ff++; while not good .
        self.failure_index_list = []
        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'],
                           blend_loc=config['comb_fake']['blend_loc'])
        self.bi = self.config['comb_fake']['bi']
        if self.bi:
            self.init_nearest()
        self.transforms=self.three_images_transform()
        self.real_choice=self.config.get('real_choice',4)
        self.domain_num=4

    def format_imglist(self,img_list):
        new_list={}
        for img,label in img_list:
            video_name = img.split('\\')[-2]
            if len(video_name)>4:
                video_name=video_name[:3]
            if video_name in new_list:
                new_list[video_name].append(tuple((img,label)))
            else:
                new_list[video_name]=[]
                new_list[video_name].append(tuple((img,label)))
        for key in new_list.keys():
            new_list[key].sort()
        return new_list

    def reorder_landmark(self, landmark):
        landmark = landmark.copy()  # 创建landmark的副本
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def three_images_transform(self):
        if self.bi:
            ad_target = {f'image1': 'image', 'image2': 'image','image3': 'image'}
        else:
            ad_target = {f'image1': 'image', 'image2': 'image'}
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets=ad_target,
            p=1.)


    def index_reformat(self,index):
        video,fake_frame = index // (self.config['frame_num']['train']*self.domain_num), index % (self.config['frame_num']['train']*self.domain_num)
        video_key=self.index_mapping[video]
        #是不是这里的real 显得尤为关键？
        real_frame = fake_frame % self.real_choice
        return video_key,fake_frame,real_frame

    def remove_border_mask(self,mask,deepfake_img,real_img):
        pass

    def colorTransfer(self, src, dst, mask):
        transferredDst = np.copy(dst)
        maskIndices = np.where(mask != 0)
        maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.float32)
        maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.float32)

        # Compute means and standard deviations
        meanSrc = np.mean(maskedSrc, axis=0)
        stdSrc = np.std(maskedSrc, axis=0)
        meanDst = np.mean(maskedDst, axis=0)
        stdDst = np.std(maskedDst, axis=0)

        # Perform color transfer
        maskedDst = (maskedDst - meanDst) * (stdSrc / stdDst) + meanSrc
        maskedDst = np.clip(maskedDst, 0, 255)

        # Copy the entire background into transferredDst
        transferredDst = np.copy(dst)
        # Now apply color transfer only to the masked region
        transferredDst[maskIndices[0], maskIndices[1]] = maskedDst.astype(np.uint8)

        return transferredDst

    def randaffine(self, img, mask):
        f = A.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = A.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def dynamic_blend(self,source, target, mask):
        mask_blured = get_blend_mask(mask)
        # 这里blend ratio确实值得考虑。原文里应该是1  #0.25, 0.5, 0.75,
        if self.config['soft_bi']:
            blend_list = [0.25, 0.5, 0.75,1, 1, 1]
        else:
            blend_list = [1, 1, 1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
        mask_blured *= blend_ratio
        img_blended = (mask_blured * source + (1 - mask_blured) * target)
        return img_blended, mask_blured

    def two_blending(self, img_bg, img_fg, mask=None,landmark=None):
        if mask is None:
            if np.random.rand() < 0.25:
                landmark = landmark[:68]
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img_bg,
                                   hull_type=(random.choice([0,1,2]) if self.config['comb_fake']['blend_loc']=='trick_whole_face' else None))

            logging.disable(logging.NOTSET)
        source = img_fg.copy()
        target = img_bg.copy()
        source_v2, mask_v2 = self.randaffine(source, mask)
        source_v3=self.colorTransfer(target,source_v2,mask_v2)
        img_blended, mask = self.dynamic_blend(source_v3, target, mask_v2)
        img_blended = img_blended.astype(np.uint8)
        img = img_bg.astype(np.uint8)

        return img, img_blended, mask.squeeze(2)

    def init_nearest(self):
        if os.path.exists('nearest_face_info_worep.pkl'):
            with open('nearest_face_info_worep.pkl', 'rb') as f:
                face_info = pickle.load(f)

        for key,val in face_info.items():
            key_video = key[:-7]
            for each in val:
                each_video = each[:-7]
                if key_video == each_video:
                    face_info[key].remove(each)
        self.face_info = face_info
        # Check if the dictionary has already been created
        if os.path.exists('landmark_dict_new.pkl'):
            with open('landmark_dict_new.pkl', 'rb') as f:
                landmark_dict = pickle.load(f)
        self.landmark_dict = landmark_dict

    def generate_bi(self,real_landmark_path,real_image,mask,):
        landmark_path_fg, real_landmark_path = self.get_fg_bg(real_landmark_path)
        image_path_fg = landmark_path_fg.replace('landmarks', 'frames').replace('.npy', '.png')
        image_fg = self.load_rgb(image_path_fg)
        image_fg = np.array(image_fg)  # Convert to numpy array for data augmentation
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        #还是重新造bi的好
        real_image, bi_image, mask_f = self.two_blending(real_image.copy(), image_fg.copy(),None , landmark.copy()) #mask.squeeze(-1).copy()
        return real_image, bi_image, mask_f

    def get_fg_bg(self, one_lmk_path):
        """
        Get foreground and background paths
        """
        bg_lmk_path = one_lmk_path
        # Randomly pick one from the nearest neighbors for the foreground
        if bg_lmk_path in self.face_info:
            fg_lmk_path = random.choice(self.face_info[bg_lmk_path][:self.config['bi_topN']])
        else:
            fg_lmk_path = bg_lmk_path
        return fg_lmk_path, bg_lmk_path


    def __getitem__(self, index):
        video_key, fake_frame, real_frame = self.index_reformat(index)
        try:
            # Get the real image paths and labels
            fake_image_path, fake_label = self.fake_imgdict[video_key][fake_frame] # if (fake_frame<8 or (fake_frame>16 and fake_frame<24)) else fake_frame-8]
            # real_index= int(index// (len(self.fake_imglist) // len(self.real_imglist) ))-2
            # # real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
            real_image_path, real_label = self.real_imgdict[video_key][real_frame]
        except Exception as e:
            # Get the real image paths and labels
            fake_image_path, fake_label = self.fake_imgdict[video_key][
                min(fake_frame, len(self.fake_imgdict[video_key]) - 1)]
            # real_index= int(index// (len(self.fake_imglist) // len(self.real_imglist) ))-2
            # # real_index = random.randint(0, len(self.real_imglist) - 1)  # Randomly select a real image
            real_image_path, real_label = self.real_imgdict[video_key][
                min(real_frame, len(self.real_imgdict[video_key]) - 1)]
            print(f"fail to load index: {index} {video_key} {fake_frame} {real_frame}")
        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        # Load the fake and real images
        fake_image = self.load_rgb(fake_image_path)
        real_image = self.load_rgb(real_image_path)

        deepfake_image = np.array(fake_image)  # Convert to numpy array for data augmentation
        real_image = np.array(real_image)  # Convert to numpy array for data augmentation

        blendfake_image, real_image, blend_mask = self.sbi(real_image, landmark,aug=False,return_mask=True)
        if self.bi:
            real_image, bi_image, mask_f = self.generate_bi(real_landmark_path,real_image.copy(), blend_mask.copy())

        if self.mode == 'train':
            if not self.bi:
                transformed = \
                    self.transforms(image=blendfake_image.astype('uint8'), image1=real_image.astype('uint8'), image2=deepfake_image.astype('uint8'))
                blendfake_image, real_image, deepfake_image = transformed['image'],transformed['image1'],transformed['image2']
            else:
                transformed = \
                    self.transforms(image=blendfake_image.astype('uint8'), image1=real_image.astype('uint8'), image2=deepfake_image.astype('uint8'),image3=bi_image.astype('uint8'))
                blendfake_image, real_image, deepfake_image, bi_image = transformed['image'], transformed['image1'], transformed['image2'], transformed['image3']
        if fake_image is None:
            blendfake_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = self.config['comb_fake']['blendfake_label']

        # To tensor and normalize for fake and real images
        blendfake_image_trans = self.normalize(self.to_tensor(blendfake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))
        deepfake_image_trans = self.normalize(self.to_tensor(deepfake_image))
        if self.bi:
            bi_image_trans = self.normalize(self.to_tensor(bi_image))
        else:
            bi_image_trans = None
        return {
            "bifake": (bi_image_trans,self.config['comb_fake'].get('bifake_label',3)),
            "blendfake": (blendfake_image_trans, fake_label),
            "deepfake": (deepfake_image_trans, self.config['comb_fake']['deepfake_label']),
            "real": (real_image_trans, real_label)
        }

    def __len__(self):
        return len(self.fake_imgdict)*self.config['frame_num']['train']*self.domain_num





    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        global c

        # Separate the image, label, landmark, and mask tensors for fake and real data
        blendfake_images, blendfake_labels = zip(*[data["blendfake"] for data in batch])
        deepfake_images, deepfake_labels = zip(*[data["deepfake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])
        bifake_images,bifake_labels = zip(*[data["bifake"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        blendfake_images = torch.stack(blendfake_images, dim=0)
        blendfake_labels = torch.Tensor(blendfake_labels)
        deepfake_images = torch.stack(deepfake_images, dim=0)
        deepfake_labels = torch.Tensor(deepfake_labels)
        if bifake_images[0] is not None:
            bifake_images = torch.stack(bifake_images, dim=0)
            bifake_labels = torch.Tensor(bifake_labels)
        else:
            bifake_images = deepfake_images[0:0]
            bifake_labels = deepfake_labels[0:0]

        select_fake_images,select_fake_label = select_fake(blendfake_images,deepfake_images,bifake_images,blendfake_labels,deepfake_labels,bifake_labels,
                                                     bi_ratio=c['comb_fake'].get('bifake_ratio',0),b_ratio=c['comb_fake']['blendfake_ratio'],df_ratio=c['comb_fake']['deepfake_ratio'],order=c['comb_fake']['order'])
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.Tensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, select_fake_images], dim=0)
        labels = torch.cat([real_labels, select_fake_label], dim=0)
        
        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
        }
        return data_dict


def save_mid(tensors,idx):
    root=f'sbiplus_img/four_align_imgs_v2/train/{idx}'
    os.makedirs(root,exist_ok=True)
    label = ['real','sbi','bi','deepfake']
    real_img = torchvision.transforms.ToPILImage()(tensors[0]*0.5+0.5)
    sbi_img = torchvision.transforms.ToPILImage()(tensors[1]*0.5+0.5)
    bi_img = torchvision.transforms.ToPILImage()(tensors[2]*0.5+0.5)
    df_img = torchvision.transforms.ToPILImage()(tensors[3]*0.5+0.5)
    real_img.save(f'{root}/0_real.png')
    sbi_img.save(f'{root}/1_sbi.png')
    bi_img.save(f'{root}/2_bi.png')
    df_img.save(f'{root}/3_df.png')

if __name__ == '__main__':
    with open(r'H:\code\DeepfakeBench\training\config\detector\sbiplus_effnb4_v9.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(r'H:\code\DeepfakeBench\training\config\train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])

    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['test_dataset']='FaceForensics++'
    config.update(config2)
    config['soft_bi']=False
    config['frame_num']['train']=8
    config['frame_num']['test']=8
    train_set = SBIPlusV2Dataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        # save_mid(batch['image'],iteration)
        if iteration > 1000:
            break