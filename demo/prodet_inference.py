


import torch
import torch.nn as nn
from networks import EfficientNetB4
# from detectors import DETECTOR


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FeatureAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_conv1 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8))
        )
        self.inverse_conv2 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8))
        )
        self.inverse_conv3 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8))
        )
        self.attention = nn.Sequential(
            nn.Conv2d(512*4, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # here are two types: same size or 1 size
            nn.Sigmoid(),
        )

    def forward(self,x1,x2,x3,feature):
        x1 = self.inverse_conv1(x1)
        x2 = self.inverse_conv2(x2)
        x3 = self.inverse_conv3(x3)
        att = self.attention(torch.cat((x1,x2,x3,feature),dim=1))
        feature = feature*att
        return feature

# @DETECTOR.register_module(module_name='Prodet_infer')
class ProDet_infer(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.backbone = self.build_backbone()
        
    def build_backbone(self):
        # prepare the backbone
        model_config = {'mode': 'original', 'num_classes': 2, 'inc': 3, 'dropout': False, 'pretrained': None}
        backbone = EfficientNetB4(model_config)
        self.adjust_feature = nn.Conv2d(1792, 512, 1)
        self.blend_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.fea_att = FeatureAttentionBlock()
        self.ID_inconsistency_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.deepfake_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        return backbone
    
    def features(self, data_dict: dict) -> torch.tensor:
        feature_raw = self.backbone.features(data_dict['image'])
        x = self.adjust_feature(feature_raw)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        df_pred=self.deepfake_classifier(features)
        bld_pred = self.blend_classifier(features)
        bi_pred = self.ID_inconsistency_classifier(features)
        return df_pred,bld_pred,bi_pred


    def forward(self, data_dict: dict,inference=False) -> dict:
        pred_dict={}
        # input: real blend bi fake

        # get the features by backbone
        features = self.features(data_dict)

        # get the prediction by each classifier
        df_pred,bld_pred,bi_pred = self.classifier(features)

        att_features = self.fea_att(df_pred,bld_pred,bi_pred,features)
        pred = self.final_classifier(att_features)

        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict.update({'cls': pred, 'prob': prob})
        return pred_dict

if __name__ == '__main__':

    detector=ProDet_infer().cuda()
    ckpt_path = 'training/weights/ProDet_best.pth'
    ckpt=torch.load(ckpt_path)
    detector.load_state_dict(ckpt,strict=False)
    '''
    Your inference code:
    datadict={}
    datadict['image']=image
    res=detector(datadict)
    '''
