import os
import torch
import logging
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.utils import *
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset_img import Fusion

from fvcore.nn import FlopCountAnalysis
from common.graph_utils import adj_mx_from_skeleton
from datetime import datetime
import pytz

opt = opts().parse()
exec('from model.' + opt.model + ' import Model')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.cuda.set_device(f'cuda:{opt.gpu}')

def train(opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight=None):
    return step('train', opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight)

def val(opt, actions, val_loader, model, writer):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, writer)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, writer=None, adaptive_weight=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    TQDM = tqdm(enumerate(dataLoader), total=len(dataLoader), ncols=100)
    for i, data in TQDM:
        batch_cam, gt_3D, input_2D, image, action, subject, scale, bb_box, start, end = data

        [input_2D, image, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, image, gt_3D, batch_cam, scale, bb_box])

        if split == 'train':
            output_3D = model(input_2D, image)

        elif split == 'test':
            input_2D, output_3D = input_augmentation(input_2D, image, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            loss = mpjpe_cal(output_3D, out_target)

            TQDM.set_description(f'Epoch [{epoch}/{opt.nepoch}]')
            TQDM.set_postfix({"l": loss.item()})

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer is not None:
                step_num = epoch * len(dataLoader) + i
                writer.add_scalar("Loss/Total_Loss", loss.item(), step_num)

        elif split == 'test':
            if output_3D.shape[1] != 1:
                output_3D = output_3D[:, opt.pad].unsqueeze(1)
            output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2

def input_augmentation(input_2D, image, model):
    input_2D_non_flip = input_2D[:, 0]
    output_3D_non_flip = model(input_2D_non_flip, image)

    return input_2D_non_flip, output_3D_non_flip

def count_parameters_in_M(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def count_flops_in_G(model, input_2d, input_img):
    flops = FlopCountAnalysis(model, (input_2d, input_img))
    return flops.total() / 1e9

def count_used_parameters_cuda(model, example_input_2d, example_img):
    model.eval()
    used_params = set()

    def hook_fn(module, input, output):
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                used_params.add(param.data_ptr())

    hooks = []
    for module in model.modules():
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(example_input_2d, example_img)

    for h in hooks:
        h.remove()

    used_param_count = sum(p.numel() for p in model.parameters() if p.data_ptr() in used_params)
    return used_param_count / 1e6

if __name__ == '__main__':

    root_path = opt.root_path
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    model = Model(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=opt.embed_dim_ratio, img_embed_dim_ratio=opt.img_embed_dim_ratio,
                  depth=opt.depth, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.)

    gpu_ids = list(map(int, opt.gpu.split(',')))
    if len(gpu_ids) == 1:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
        model = model.to(device)
    else:

        print(f"Let's use {len(gpu_ids)} GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(f"cuda:{gpu_ids[0]}")

    model_dict = model.state_dict()
    if opt.previous_dir != '':
        print('pretrained model path:', opt.previous_dir)
        model_path = opt.previous_dir

        pre_dict = torch.load(model_path)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())

    optimizer = optim.AdamW(all_param, lr=lr, weight_decay=0.1)

    local_tz = pytz.timezone("Asia/Shanghai")
    current_time = datetime.now(local_tz).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'runs/{opt.model}_{current_time}'
    writer = SummaryWriter(log_dir)

    flag = 0
    best_epoch = 0

    for epoch in range(1, opt.nepoch + 1):
        if opt.train:

            loss = train(opt, actions, train_dataloader, model, optimizer, epoch, writer)

        p1, p2 = val(opt, actions, test_dataloader, model, writer)

        if opt.train and p1 < opt.previous_best_threshold:
            best_epoch = epoch
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
            opt.previous_best_threshold = p1

        if opt.train == 0:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)
