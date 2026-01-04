import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
import torch.nn.functional as F

part_indices = {
    'left_arm':  [11, 12, 13],
    'right_arm': [14, 15, 16],
    'left_leg':  [4, 5, 6],
    'right_leg': [1, 2, 3],
    'torso':     [0, 7, 8, 9, 10],
}

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def mpjpe_cal(predicted, target):

    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted for input shape (batch_size, frame, joints, 3).
    """
    assert predicted.shape == target.shape, "Predicted and target shapes must match"

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=-1, keepdim=True), dim=-2, keepdim=True)

    norm_target = torch.mean(torch.sum(target * predicted, dim=-1, keepdim=True), dim=-2, keepdim=True)

    scale = norm_target / norm_predicted

    return mpjpe_cal(scale * predicted, target)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def depth_uncertainty_loss(pred_depth, pred_sigma, gt_depth):
    """
    计算深度不确定性损失
    :param pred_depth: (b, f, j) - 预测的深度均值 μ
    :param pred_sigma: (b, f, j) - 预测的深度标准差 σ
    :param gt_depth: (b, f, j) - 真实深度 z
    """

    pred_sigma = F.softplus(pred_sigma) + 1e-6
    loss = ((gt_depth - pred_depth) ** 2) / (2 * pred_sigma ** 2) + torch.log(pred_sigma)

    if torch.any(loss < 0):
        print("Negative loss detected. pred_sigma values:")
        print(pred_sigma)
    return loss.mean()

def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()

def nig_var(gamma, v, alpha, beta, y):
    var = torch.abs(beta / (v * (alpha - 1)))
    return var.mean()

def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1.0):
    L1 = nig_var(*dist_params, y)
    L2 = nig_reg(*dist_params, y)
    return L1 + lamb * L2

def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

    return action_error_sum

def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum

def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    normX = np.where(normX == 0, 1e-10, normX)
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)

def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
                          {'p1': AccumLoss(), 'p2': AccumLoss()}
                      for i in range(len(actions))})
    return error_sum

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2

def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg

def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name

def save_model_step1_best(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存 Step1（joint training）最优模型，命名沿用原始规则，增加 step1 前缀。
    """
    file_path = '%s/step1_model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    if previous_name and os.path.exists(previous_name):
        os.remove(previous_name)
    torch.save(model.state_dict(), file_path)
    return file_path

def save_model_step2_best(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存 Step2（meta training）最优模型，命名沿用原始规则，增加 step2 前缀。
    """
    file_path = '%s/step2_model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    if previous_name and os.path.exists(previous_name):
        os.remove(previous_name)
    torch.save(model.state_dict(), file_path)
    return file_path

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))

def define_adaptive_weight():
    train_list = ['S1', 'S5', 'S6', 'S7', 'S8']
    actions = ['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating', 'Greeting 1',
               'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1',
               'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1',
               'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether',
               'Directions 2', 'Discussion 2', 'Discussion 3', 'Eating 1', 'Greeting 2', 'Photo 2', 'Sitting',
               'SittingDown 1', 'Waiting 2', 'Posing 2', 'Waiting 3', 'Phoning 2', 'Walking 2', 'WalkTogether 2']
    act_dict = {}
    for act in actions:
        act_dict[act] = np.ones(6400)

    adaptive_weight = {}
    for sbj in train_list:
        adaptive_weight[sbj] = act_dict

    return adaptive_weight

def get_adaptive_weight(adaptive_weight, subject, action, start, end):
    N = len(subject)
    weights = torch.zeros((1, N))
    for idx in range(N):
        weights[0, idx] = sum(adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]]) / (end[idx] - start[idx])
    return weights

def fil_ex(se, min=0.1, max=0.9):
    se_num = se.shape[0]
    se_sort = np.sort(se)
    se_sort_file = se_sort[int(se_num * min):int(se_num * max)]
    mean, var = np.mean(se_sort_file), np.sqrt(np.var(se_sort_file))

    if var < 2:
        return mean, var
    else:
        return mean, torch.tensor(1e-10)

def update_adaptive_weight(adaptive_weight, subject, action, start, end, loss_batch):
    N = len(subject)
    loss_batch_for_mean_var = loss_batch.detach().cpu().numpy()
    mean, var = fil_ex(loss_batch_for_mean_var, min=0.05,
                       max=0.95)
    for idx in range(N):
        temp_weight = torch.exp(-(loss_batch[idx] - mean) * var).detach().cpu().numpy()
        adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]] *= temp_weight

    return adaptive_weight, mean, var

def compute_body_part_loss(pred, gt):
    """
    pred, gt: [B, 17, 3] - 3D关键点
    part_indices: dict of body parts -> list of joint indices
    loss_fn: e.g., MSELoss()
    """

    loss = 0.0
    for part, idx in part_indices.items():
        pred_part = pred[:, idx, :]
        gt_part = gt[:, idx, :]
        loss += mpjpe_cal(pred_part, gt_part)
    return loss
