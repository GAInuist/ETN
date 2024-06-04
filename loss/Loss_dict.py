import torch.nn.functional as F
import torch


def compute_cosine_loss(attr_init):
    attr_feature_cosine = torch.einsum('bij, bjk -> bik', attr_init, attr_init.permute(0, 2, 1))
    loss_cos = F.mse_loss(attr_feature_cosine,
                          torch.eye(attr_feature_cosine.shape[1]).unsqueeze(dim=0).repeat(attr_feature_cosine.shape[0],
                                                                                          1, 1).cuda(),
                          reduction='mean')
    return loss_cos


def compute_reg_loss(package, class_prototype, bias_local):
    attn_mask = package['delta_l']
    loss_reg = F.mse_loss(attn_mask + bias_local, class_prototype)
    return loss_reg


def get_semantic_loss(package, bias_local, bias_global, attribute):
    a_local = attribute - bias_local
    a_global = attribute - bias_global
    local_results, global_results = package['local_result'], package['global_result']
    local_loss = F.mse_loss(local_results, a_local, reduction='mean')
    global_loss = F.mse_loss(global_results, a_global, reduction='mean')
    loss_cons = local_loss + global_loss
    return loss_cons
