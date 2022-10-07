from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.lr_schedule import inv_lr_scheduler
from termcolor import colored
from utils.return_dataset import return_dataset_DARK
from utils.loss import CrossEntropy_SL_LS
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from log_utils.utils import ReDirectSTD

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='Distilling_and_refining_for_SSDA')

# Training parameters
parser.add_argument('--steps', type=int, default=50001, metavar='N', help='maximum number of iterations to train (Default of DomainNet: 50000 / Office-home: 10000)')
parser.add_argument('--method', type=str, default='DARK')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT', help='learning rate multiplication')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Hyperparameters
parser.add_argument('--T', type=float, default=0.05, metavar='T', help='classifier temperature (default: 0.05)')
parser.add_argument('--ls', type=float, default=0.1, help='parameter of label smoothing')
parser.add_argument('--temperature', default=2.5, type=float, help='parameter temperature scaling')

# Checkpath settings
parser.add_argument('--checkpath_1', type=str, default='./checkpaths/RtoS_source_view', help='dir to save checkpoint')
parser.add_argument('--checkpath_2', type=str, default='./checkpaths/RtoS_target_view', help='dir to save checkpoint')
parser.add_argument('--checkpath_3', type=str, default='./checkpaths/RtoS_ensemble', help='dir to save checkpoint')
parser.add_argument('--save_check', action='store_true', default=True, help='save checkpoint or not')

# Logging settings
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model')
parser.add_argument('--log_file', type=str, default='./RtoS_3shot.log', help='dir to save checkpoint')

# Episode settings
parser.add_argument('--net', type=str, default='resnet34', help='which network to use')
parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office_home', 'visda'], help='the name of dataset')
parser.add_argument('--source', type=str, default='real', help='source domain')
parser.add_argument('--target', type=str, default='sketch', help='target domain')

# Dataloader settings
parser.add_argument('--num', type=int, default=3, help='number of labeled examples in the target')
parser.add_argument('--ways', type=int, default=10, help='number of classes sampled')
parser.add_argument('--src_shots', type=int, default=8, help='number of samples per source classes')
parser.add_argument('--trg_shots', type=int, default=3, help='number of samples per target classes')
parser.add_argument('--root', type=str, default='/media/D/Juhyun/dataset/DomainNet/', help='Dataset root')

# Resume settings
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--pre_trained', type=str, default='./pre_trained_3_shot_P2C', help='dir to save checkpoint')

args = parser.parse_args()

# For gpu
use_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

# For reproducibility
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    log_file_name = './logs/' + '/' + args.log_file
    ReDirectSTD(log_file_name, 'stdout', True)

    print('Dataset: %s, Source: %s, Target: %s, Network: %s' % (args.dataset, args.source, args.target, args.net))

    labeled_data_loader, un_target_loader_val, un_target_loader_test, unlabeled_loader, class_list = return_dataset_DARK(args)

    record_dir = 'record/%s/%s' % (args.dataset, args.method)

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # Network
    if args.net == 'resnet34':
        G = resnet34()
        inc = 512
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' not in key:
                params += [{'params': [value], 'lr': args.multi,
                            'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args.multi * 10,
                            'weight_decay': 0.0005}]

    if "resnet" in args.net:
        F1 = Predictor_deep(num_class=len(class_list), inc=inc)
        F2 = Predictor_deep(num_class=len(class_list), inc=inc)
    else:
        F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
        F2 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

    G = torch.nn.DataParallel(G.to(device))
    F1 = torch.nn.DataParallel(F1.to(device))
    F2 = torch.nn.DataParallel(F2.to(device))

    G.train()
    F1.train()
    F2.train()

    optimizer_g = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f1 = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f2 = optim.SGD(list(F2.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)


    if args.save_check:
        if not os.path.exists(args.checkpath_1):
            os.mkdir(args.checkpath_1)
        if not os.path.exists(args.checkpath_2):
            os.mkdir(args.checkpath_2)
        if not os.path.exists(args.checkpath_3):
            os.mkdir(args.checkpath_3)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f1 = []
    for param_group in optimizer_f1.param_groups:
        param_lr_f1.append(param_group["lr"])
    param_lr_f2 = []
    for param_group in optimizer_f2.param_groups:
        param_lr_f2.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(device)
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature).to(device)
    criterion_target = CrossEntropy_SL_LS(num_class=len(class_list), ls=0.1)

    labeled_data = iter(labeled_data_loader)
    len_labeled = len(labeled_data_loader)

    unlabeled_data = iter(unlabeled_loader)
    len_unlabeled = len(unlabeled_loader)

    best_acc1_val = 0.0
    best_acc2_val = 0.0
    best_ensemble_val = 0.0

    best_acc1_test = 0.0
    best_acc2_test = 0.0
    best_ensemble_test = 0.0

    for step in range(args.steps):

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f1 = inv_lr_scheduler(param_lr_f1, optimizer_f1, step, init_lr=args.lr)
        optimizer_f2 = inv_lr_scheduler(param_lr_f2, optimizer_f2, step, init_lr=args.lr)

        lr = optimizer_f1.param_groups[0]['lr']

        if step % len_labeled == 0:
            labeled_data = iter(labeled_data_loader)

        if step % len_unlabeled == 0:
            unlabeled_data = iter(unlabeled_loader)

        data_l = next(labeled_data)
        data_unlabeled = next(unlabeled_data)
        data_weak_unlabeled, data_str_unlabeled = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
        data_labeled, label = data_l[0].to(device), data_l[1].long().to(device)

        num_src = args.ways * args.src_shots

        # data split
        data_s, data_t = data_labeled[:num_src], data_labeled[num_src:],
        label_s, label_t = label[:num_src], label[num_src:]

        zero_grad_all()

        # the number of each data in a batch
        ns = data_s.size(0)
        nt = data_t.size(0)
        nu = data_weak_unlabeled.size(0)

        # Source-view step ------------------------------------------------------------------------------

        data_s_view = torch.cat((data_s, data_weak_unlabeled, data_str_unlabeled), 0)

        embed_s = G(data_s_view)

        out_s = F1(embed_s)
        out_t = F2(embed_s)

        # supervision of source-view

        loss_sup_s = criterion(out_s[:ns], label_s)

        # weight for weakly augmented unlabeled target data
        pseudo_label_s = torch.softmax(out_s[ns:ns+nu].detach(), dim=-1)
        probs_topk = torch.topk(pseudo_label_s, 2, dim=-1)[0]
        weight_weak_s = torch.max(probs_topk, dim=-1)[0] - torch.min(probs_topk, dim=-1)[0]

        # weight for strongly augmented unlabeled target data
        pseudo_label_s_str = torch.softmax(out_s[ns+nu:].detach(), dim=-1)
        probs_topk = torch.topk(pseudo_label_s_str, 2, dim=-1)[0]
        weight_str_s = torch.max(probs_topk, dim=-1)[0] - torch.min(probs_topk, dim=-1)[0]

        # distilling strategy
        loss_dis_s = criterion_target(out_t[ns+nu:], pseudo_label_s, weight_weak_s.to(device)).to(device)

        # refining strategy
        loss_wcc_s = mcc_loss(out_s[ns:ns+nu], weight=weight_weak_s).to(device)
        loss_scc_s = mcc_loss(out_s[ns+nu:], weight=weight_str_s).to(device)

        # dynamic weight for scc loss
        hier_weight = 1.0-loss_wcc_s.item()

        if hier_weight < 0.0:
            hier_weight = 0.0
        elif hier_weight > 0.0:
            hier_weight = np.exp(-3.0 * (1.0 - hier_weight))

        # bridging consistency loss
        prob_weak_s = F.softmax(out_s[ns:ns+nu], dim=1)
        prob_str_s = F.softmax(out_s[ns+nu:], dim=1)
        loss_brd_s = ((weight_weak_s * weight_str_s) * ((prob_weak_s - prob_str_s) ** 2.0).sum(-1)).mean().to(device)

        loss_ref_s = loss_wcc_s + hier_weight * loss_scc_s + loss_brd_s

        # combined loss
        loss_comb_s = loss_sup_s + loss_dis_s + loss_ref_s

        loss_comb_s.backward()
        optimizer_g.step()
        optimizer_f1.step()
        optimizer_f2.step()

        zero_grad_all()

        # target-view step ------------------------------------------------------------------------------
        data_t_view = torch.cat((data_t, data_weak_unlabeled, data_str_unlabeled), 0)

        embed_t = G(data_t_view)

        out_s = F1(embed_t)
        out_t = F2(embed_t)

        # supervision of target-view
        loss_sup_t = criterion(out_t[:nt], label_t)

        # weight for weakly augmented unlabeled target data
        pseudo_label_t = torch.softmax(out_t[nt:nt+nu].detach(), dim=-1)
        probs_topk = torch.topk(pseudo_label_t, 2, dim=-1)[0]
        weight_weak_t = torch.max(probs_topk, dim=-1)[0] - torch.min(probs_topk, dim=-1)[0]

        # weight for strongly augmented unlabeled target data
        pseudo_label_t_str = torch.softmax(out_t[nt+nu:].detach(), dim=-1)
        probs_topk = torch.topk(pseudo_label_t_str, 2, dim=-1)[0]
        weight_str_t = torch.max(probs_topk, dim=-1)[0] - torch.min(probs_topk, dim=-1)[0]

        # distilling strategy
        loss_dis_t = criterion_target(out_s[nt+nu:], pseudo_label_t, weight_weak_t.to(device)).to(device)

        # refining strategy
        loss_wcc_t = mcc_loss(out_t[nt:nt+nu], weight=weight_weak_t).to(device)
        loss_scc_t = mcc_loss(out_t[nt+nu:], weight=weight_str_t).to(device)

        # dynamic weight for scc loss
        hier_weight = 1.0-loss_wcc_t.item()

        if hier_weight < 0.0:
            hier_weight = 0.0
        elif hier_weight > 0.0:
            hier_weight = np.exp(-3.0 * (1.0 - hier_weight))

        # bridging consistency loss
        prob_weak_t = F.softmax(out_t[nt:nt+nu], dim=1)
        prob_str_t = F.softmax(out_t[nt+nu:], dim=1)
        loss_brd_t = ((weight_weak_t * weight_str_t) * ((prob_weak_t - prob_str_t) ** 2.0).sum(-1)).mean().to(device)

        loss_ref_t = loss_wcc_t + hier_weight * loss_scc_t + loss_brd_t

        # combined loss
        loss_comb_t = loss_sup_t + loss_dis_t + loss_ref_t

        loss_comb_t.backward()
        optimizer_g.step()
        optimizer_f1.step()
        optimizer_f2.step()

        zero_grad_all()

        # -----------------------------------------------------------------------------------------------

        log_train = 'Source ({}) → Target ({}) | Iter: {:4d} lr: {:.4f} | loss_src_view: {:.4f}, loss_sup_s: {:.4f}, loss_sup_t: {:.4f}, weight_weak_s: {:.4f}, weight_weak_t: {:.4f}, weight_str_s: {:.4f}, weight_str_t: {:.4f}' \
            .format(args.source, args.target, step, lr, loss_comb_s.data, loss_sup_s.data, loss_sup_t.data, weight_weak_s.mean(), weight_weak_t.mean(), weight_str_s.mean(), weight_str_t.mean())

        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0 and step > 0:
            loss_val, acc_1_val, acc_2_val, correct_ensemble = test(un_target_loader_val, G, F1, F2, class_list, test=False)
            loss_test, acc_1_test, acc_2_test, correct_ensemble_test = test(un_target_loader_test, G, F1, F2, class_list, test=True)

            G.train()
            F1.train()
            F2.train()

            if acc_1_val >= best_acc1_val:
                best_acc1_val = acc_1_val
                best_acc1_test = acc_1_test

                if args.save_check:
                    print('saving model_1')
                    torch.save(G.state_dict(), os.path.join(args.checkpath_1, "G_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                    torch.save(F1.state_dict(), os.path.join(args.checkpath_1, "F1_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))

            if acc_2_val >= best_acc2_val:
                best_acc2_val = acc_2_val
                best_acc2_test = acc_2_test

                if args.save_check:
                    print('saving model_2')
                    torch.save(G.state_dict(), os.path.join(args.checkpath_2, "G_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                    torch.save(F2.state_dict(), os.path.join(args.checkpath_2, "F2_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))

            if correct_ensemble >= best_ensemble_val:
                best_ensemble_val = correct_ensemble
                best_ensemble_test = correct_ensemble_test

                print('saving model_3')
                torch.save(G.state_dict(), os.path.join(args.checkpath_3, "G_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                torch.save(F1.state_dict(), os.path.join(args.checkpath_3, "F1_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                torch.save(F2.state_dict(), os.path.join(args.checkpath_3, "F2_model_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))

            print('')
            print(colored('Best acc_1 valid %.2f, Best acc_2 valid %.2f, Best ensemble acc valid %.2f' % (best_acc1_val, best_acc2_val, best_ensemble_val), 'green'))
            print(colored('Best acc_1 test %.2f, Best acc_2 test %.2f, Best ensemble acc test %.2f' % (best_acc1_test, best_acc2_test, best_ensemble_test), 'yellow'))

            G.train()
            F1.train()
            F2.train()

        if step % args.save_interval * 10 == 0 and step > 0:
            print('saving model')


def test(loader, G, F1, F2, class_list, test=False):
    G.eval()
    F1.eval()
    F2.eval()

    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct_ensemble = 0

    size = 0
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):

            image_t, label_t = data_t

            label_t = label_t.long().to(device)
            image_t = image_t.to(device)

            feat = G(image_t)

            output1 = F1(feat)
            output2 = F2(feat)

            output_ensemble = output1 + output2

            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]

            correct1 += pred1.eq(label_t.data).cpu().sum()
            correct2 += pred2.eq(label_t.data).cpu().sum()
            correct_ensemble += pred_ensemble.eq(label_t.data).cpu().sum()

            size += label_t.size(0)

            test_loss += criterion(output1, label_t) / len(loader)

    if test:
        print(colored(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.1f}%), Accuracy C2: {}/{} ({:.1f}%), Accuracy Ensemble: {}/{} ({:.1f}%) \n'
                .format(test_loss, correct1, size, 100. * correct1 / size, correct2, size, 100. * correct2 / size, correct_ensemble, size, 100. * correct_ensemble / size), 'yellow'))
    else:
        print(colored(
            '\nValidation set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.1f}%), Accuracy C2: {}/{} ({:.1f}%), Accuracy Ensemble: {}/{} ({:.1f}%) \n'
                .format(test_loss, correct1, size, 100. * correct1 / size, correct2, size, 100. * correct2 / size, correct_ensemble, size, 100. * correct_ensemble / size), 'green'))

    return test_loss.data, 100. * float(correct1) / size, 100. * float(correct2) / size, 100. * float(correct_ensemble) / size

if __name__ == '__main__':
    main()

