import os
import errno
import logging
import numpy as np
import torch
from scipy.special import softmax
from torchvision.transforms import transforms


class AverageMeter(object):
    """class for managing loss function values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def makedir(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_transform(args):
    if args.is_train:
        trainTransform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return trainTransform
    else:
        testTransform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return testTransform


def logging_info_flow(metric_dict, epoch, loss_item):
    H, U, S = metric_dict['H'], metric_dict['gzsl_unseen'], metric_dict['gzsl_seen']
    logging.info(f'Num_epoch:{epoch}\nLoss: {loss_item}  H:{H},  U:{U},  S:{S}\n')


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def get_reprs(model, data_loader, args, w2v):
    model.eval()
    reprs, local_result, global_result = [], [], []
    for _, (data, _) in enumerate(data_loader):
        data = data.cuda()
        with torch.no_grad():
            result_dict = model(data, w2v)
            feat_l, feat_g = result_dict['local_result'], result_dict['global_result']
            bias_l, bias_g = result_dict['local_bias'], result_dict['global_bias']
            feat_l_bias, feat_g_bias = feat_l + bias_l, feat_g + bias_g
            v_prototype = args.beta1 * feat_l_bias + args.beta2 * feat_g_bias

        reprs.append(v_prototype.cpu().data.numpy())
        local_result.append(feat_l_bias.cpu().data.numpy())
        global_result.append(feat_g_bias.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)
    return reprs


def validation(model, seen_loader, seen_labels, unseen_loader, unseen_labels, attrs_mat, args, w2v):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, seen_loader, args=args, w2v=w2v)
        unseen_reprs = get_reprs(model, unseen_loader, args=args, w2v=w2v)

    # Labels
    uniq_labels = np.unique(np.concatenate([seen_labels, unseen_labels]))
    updated_seen_labels = np.searchsorted(uniq_labels, seen_labels)
    uniq_updated_seen_labels = np.unique(updated_seen_labels)
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]

    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ trunc_attrs_mat.T, axis=1)
    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ trunc_attrs_mat.T, axis=1)

    gammas = np.arange(0.0, 1.1, 0.1)
    gamma_opt = 0
    H_max = 0
    gzsl_seen_acc_max = 0
    gzsl_unseen_acc_max = 0
    # Calibrated stacking
    for igamma in range(gammas.shape[0]):
        # Calibrated stacking
        gamma = gammas[igamma]
        gamma_mat = np.zeros(trunc_attrs_mat.shape[0])
        gamma_mat[uniq_updated_seen_labels] = gamma

        gzsl_seen_pred_labels = np.argmax(gzsl_seen_sim - gamma_mat, axis=1)
        gzsl_seen_acc = compute_accuracy(gzsl_seen_pred_labels, updated_seen_labels, uniq_updated_seen_labels)

        gzsl_unseen_pred_labels = np.argmax(gzsl_unseen_sim - gamma_mat, axis=1)
        gzsl_unseen_acc = compute_accuracy(gzsl_unseen_pred_labels, updated_unseen_labels, uniq_updated_unseen_labels)

        H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

        if H > H_max:
            gzsl_seen_acc_max = gzsl_seen_acc
            gzsl_unseen_acc_max = gzsl_unseen_acc
            H_max = H
            gamma_opt = gamma

    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc_max * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc_max * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H_max * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma_opt))
    return gamma_opt


def test_GZSL(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, gamma,
              args, w2v):
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # Representation
    with torch.no_grad():
        seen_reprs_eve = get_reprs(model, test_seen_loader, args=args, w2v=w2v)
        unseen_reprs_eve = get_reprs(model, test_unseen_loader, args=args, w2v=w2v)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes every
    gzsl_seen_sim = softmax(seen_reprs_eve @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)
    gzsl_unseen_sim = softmax(unseen_reprs_eve @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)
    H_eve = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    package = {'H': H_eve * 100, 'gzsl_unseen': gzsl_unseen_acc * 100,
               'gzsl_seen': gzsl_seen_acc * 100}
    print(f'GZSL Seen: averaged per-class accuracy: {gzsl_seen_acc * 100}')
    print(f'GZSL Unseen: averaged per-class accuracy: {gzsl_unseen_acc * 100}')
    print(f'GZSL: harmonic mean (H): {H_eve * 100}')
    print('GZSL: gamma: {0:.2f}'.format(gamma))
    return package
