import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None, test=False):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if test == True:
                cls_score = self.classifier(feat_cls)
                return cls_score
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

        """ def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path)) """
        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')

        # 兼容 {'state_dict': ...}
        if isinstance(param_dict, dict) and 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        loaded, skipped = 0, 0
        model_sd = self.state_dict()

        for i in param_dict:
            # 1) 只加载 student 的
            if 'student.' not in i:
                continue

            try:
                key = i.replace('module.', '').replace('student.', '')

                # （可选但很稳）避免 shape 不匹配导致 silent skip
                if key in model_sd and model_sd[key].shape == param_dict[i].shape:
                    model_sd[key].copy_(param_dict[i])
                    loaded += 1
                else:
                    skipped += 1
            except:
                skipped += 1
                continue

        print('Loading pretrained STUDENT model from {}'.format(trained_path))
        print(f'Loaded: {loaded}, Skipped: {skipped}')


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

class SinkhornLoss(nn.Module):
    def __init__(self, blur, n_iters=100, OT_LOSS_WEIGHT=0.001,FGW_WEIGHT=0.5,beta=0.1,gamma=0.01):
        """
        初始化Sinkhorn loss计算器。

        参数:
        - reg: 正则化系数，用于平滑Sinkhorn-Knopp算法。
        - max_iters: 最大迭代次数。
        - tol: 收敛容差。
        """
        super(SinkhornLoss, self).__init__()
        self.blur = blur
       
        self.n_iters = n_iters  # 2: 1 : 10
        self.FGW_WEIGHT = FGW_WEIGHT
        self.OT_LOSS_WEIGHT = OT_LOSS_WEIGHT
        self.beta = beta
        self.gamma = gamma
        self.alpha = 0.5
        self.eps = 1e-5
        self.spatial_grid = 1

    def forward(self, x, y ,ids):
        if self.OT_LOSS_WEIGHT == 0:
            loss = torch.tensor(0) 
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)  # 对每个特征向量归一化
            y = torch.nn.functional.normalize(y, p=2, dim=1)  # 对每个特征向量归一化 
            loss, _ = fused_gromov_wasserstein_loss(y, x, alpha=self.FGW_WEIGHT,inner_sinkhorn_iter=self.n_iters,reg=self.blur)
        return loss *  self.OT_LOSS_WEIGHT

def sinkhorn_torch(p, q, C, reg, numIter=100, eps=1e-8):
    """
    使用 Sinkhorn 算法求解带熵正则化的最优传输问题：
      T = argmin_{T in U(p,q)} <T, C> - reg * H(T)
      
    参数：
      p: 目标分布，shape (n1,)
      q: 源分布，shape (n2,)
      C: 代价矩阵，shape (n1, n2)
      reg: 熵正则化参数
      numIter: Sinkhorn 迭代次数
      eps: 防止除零的小常数
      
    返回：
      T: 最优传输矩阵，shape (n1, n2)
    """
    n1, n2 = C.shape
    # 计算核矩阵 K = exp(-C/reg)
    K = torch.exp(-C / reg)
    #print("K", K)
    # 初始化标量因子
    u = torch.ones(n1, device=C.device) / n1
    v = torch.ones(n2, device=C.device) / n2
    
    for i in range(numIter):
        u = p / (K @ v + eps)
        v = q / (K.t() @ u + eps)
    T = torch.diag(u) @ K @ torch.diag(v)
    #print("T", T)
    return T

def fused_gromov_wasserstein_loss(X, Y, alpha=0.5, outer_iter=20, inner_sinkhorn_iter=100, reg=0.1):
    """
    计算两个域之间的 Fused Gromov-Wasserstein (FGW) loss，
    并返回最优传输矩阵 T 。
    
    参数：
      X: Tensor，shape (n1, d)，教师模型提取的特征（干净图）
      Y: Tensor，shape (n2, d)，学生模型提取的特征（降质图）
      alpha: 平衡参数，控制特征匹配与结构匹配的权重，
             当 alpha=0 时仅匹配特征距离，alpha=1 时仅匹配结构关系。
      outer_iter: 外层迭代次数，用于更新传输矩阵 T
      inner_sinkhorn_iter: Sinkhorn 迭代次数（每次更新 T 时使用）
      reg: Sinkhorn 熵正则化参数
      
    返回：
      loss: FGW loss 标量
      T: 最优传输矩阵，shape (n1, n2)
    """
    n1 = X.shape[0]
    n2 = Y.shape[0]
    device = X.device
    p = torch.ones(n1, device=device) / n1
    q = torch.ones(n2, device=device) / n2
    
    C1 = torch.cdist(X, X, p=2)**2
    C2 = torch.cdist(Y, Y, p=2)**2
    M = torch.cdist(X, Y, p=2)**2

    
    # 构造四维张量L，注意避免直接展开时内存过大
    C1_exp = C1.unsqueeze(2).unsqueeze(3)  # (n1, n1, 1, 1)
    C2_exp = C2.unsqueeze(0).unsqueeze(1)  # (1, 1, n2, n2)
    L_tensor = (C1_exp - C2_exp)**2  # (n1, n1, n2, n2)

    T = torch.outer(p, q)
    
    for it in range(outer_iter):
        A = torch.einsum('ijkl,jl->ik', L_tensor, T)
        
        G = (1 - alpha) * M + alpha * A  # 保持G的构造不变
        
        T = sinkhorn_torch(p, q, G, reg, numIter=inner_sinkhorn_iter)
      
    
    A = torch.einsum('ijkl,jl->ik', L_tensor, T)
    
    loss = (1 - alpha) * torch.sum(T * M) + alpha * torch.sum(T * A)
    return loss, T


class combine_teacher_student(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, __factory_T_type):
            super(combine_teacher_student, self).__init__()
            # last_stride = cfg.MODEL.LAST_STRIDE

            self.teacher = build_transformer(num_classes, camera_num, view_num, cfg, __factory_T_type)
            self.student = build_transformer(num_classes, camera_num, view_num, cfg, __factory_T_type)

            teacher_model_path = cfg.MODEL.TEACHER_WEIGHT
            student_model_path = cfg.MODEL.STUDENT_WEIGHT

           
            self.teacher.load_param(teacher_model_path)
            if student_model_path != "":
                self.student.load_param(student_model_path)

            self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
            # 冻结teacher模型的参数
            for param in self.teacher.parameters():
                param.requires_grad = False 
            
            blur = cfg.MODEL.OT_BLUR
            # self.DistillationLoss = DistillationLoss(blur)
            n_iters = cfg.MODEL.N_ITERS
            FGW_WEIGHT = cfg.MODEL.FGW_WEIGHT
            OT_LOSS_WEIGHT = cfg.MODEL.OT_LOSS_WEIGHT
            self.OT_LOSS_WEIGHT = OT_LOSS_WEIGHT
            self.DistillationLoss = SinkhornLoss(blur, n_iters=n_iters,OT_LOSS_WEIGHT=OT_LOSS_WEIGHT, FGW_WEIGHT=FGW_WEIGHT)

            self.in_planes = self.student.in_planes
  
            self.device = "cuda"
    
    def forward(self, x, y, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        if self.training:
            img_haze = x
            img_clean = y
            if  self.OT_LOSS_WEIGHT != 0:
                _, feat_clean = self.teacher(img_clean,cam_label=cam_label, view_label=view_label)
            else:
                feat_clean = 0
            cls_score, feat_haze = self.student(img_haze,cam_label=cam_label, view_label=view_label)
            
            OTloss = self.DistillationLoss(feat_haze, feat_clean, label)

            return cls_score, feat_haze, OTloss  # global feature for triplet loss

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    elif cfg.MODEL.NAME == 'teacher_OT':
        model = combine_teacher_student(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building CD2===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
