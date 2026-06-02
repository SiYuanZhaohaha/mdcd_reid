import os
from config import cfg
import argparse
from datasets import make_dataloader, make_dataloader_new
from utils.logger import setup_logger
import torch
import torch.nn as nn
from model import make_model

def compare_models(model_a, model_b):
    # 获取两个模型的状态字典
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    
    # 遍历模型A的参数，检查在模型B中是否存在并计算差异
    for name, param_a in state_dict_a.items():
        if name in state_dict_b:
            param_b = state_dict_b[name]
            # 计算平均绝对差值
            diff = (param_a - param_b).abs().mean().item()
            print(f"Layer {name}: average parameter difference = {diff}")
        else:
            print(f"Layer {name} not found in the second model.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    """ (train_loader, train_loader_normal, val_loader, corrupted_val_loader, corrupted_query_loader, corrupted_gallery_loader,
     num_query, num_classes, camera_num, view_num) = make_dataloader(cfg) """
  
        # train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader_fog(cfg)
    (train_loader, _, val_loader, corrupted_val_loader,corrupted_query_loader, corrupted_gallery_loader,
        num_query, num_classes, camera_num, view_num) = make_dataloader_new(cfg)

    loader_list = [
            val_loader, corrupted_val_loader, corrupted_query_loader,
            corrupted_gallery_loader
    ]
    name = [
        "Clean eval", "Corrupted eval", "Corrupted query",
        "Corrupted gallery"
        ]
    # num_classes = 751
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num )
    model.load_param(cfg.TEST.WEIGHT)
    device = 'cuda'
    model.to(device)
    model.eval()  # 将模型设为评估模式

    # 初始化一个字典来存储每个参数的 Fisher 信息（只计算对角部分）
    fisher_dict = {
        name: torch.zeros_like(param, device=device) 
        for name, param in model.named_parameters() if param.requires_grad
    }

    total_samples = 0
    for n_iter, (img_clean, img_fog, vid, target_cam, target_view) in enumerate(train_loader):
    # 遍历干净数据集
        # 将输入和标签移至设备
        #print("11")
        img_clean = img_clean.to(device)
        vid = vid.to(device)
        target_cam = target_cam.to(device)
        target_view = target_view.to(device)
        # 清零梯度
        model.zero_grad()
        
        # 前向传播
        cls_score = model(img_clean, cam_label=target_cam, view_label=target_view, test=True)
        #print("vid",vid)
        # 计算交叉熵损失（可根据实际情况选择其他损失函数）
        loss = nn.functional.cross_entropy(cls_score, vid)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 获取当前 batch 的样本数
        batch_size = img_clean.size(0)
        total_samples += batch_size
        
        # 累计每个参数的梯度平方值
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 注意：乘以 batch_size 以累计每个样本的贡献
                fisher_dict[name] += (param.grad.detach() ** 2) * batch_size

    # 对累积结果进行归一化，得到每个参数的平均梯度平方（即 Fisher 信息的估计）
    for name in fisher_dict:
        fisher_dict[name] /= total_samples
    torch.save(fisher_dict, cfg.DATASETS.NAMES + "fisher_dict_TransReIDSSL.pth")
    # 现在 fisher_dict 中每个 key 对应的 tensor 就是该参数的 Fisher 信息估计
    print("Fisher 信息计算完毕！")
    