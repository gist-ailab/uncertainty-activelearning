from tabnanny import check
import torch
from dataset.data import *
import torch.nn.functional as F
from util.meter import *
from dataset.semi_data import *
from util.torch_dist_sum import *
from util.std_convert import std_convert
import argparse
from util.accuracy import accuracy
from util.dist_init import *
from network.google_wide_resnet import wide_resnet28w2, wide_resnet28w8
from network.resnet import resnet18
from network.head import *
import time
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from network.simmatch import SimMatch
import os
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--episodes', type=int, default=8)
parser.add_argument('--checkpoint', type=str, default='100label.pt')
parser.add_argument('--label_per_class', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--bank_m', type=float, default=0.7)
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--c_smooth', type=float, default=0.9)
parser.add_argument('--lambda_in', type=float, default=1)
parser.add_argument('--st', type=float, default=0.1)
parser.add_argument('--tt', type=float, default=0.1)
parser.add_argument('--load', type=bool, default=False)

args = parser.parse_args()
print(args)


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

epochs = args.epochs

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, local_rank, rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_x = AverageMeter('X', ':.4e')
    losses_u = AverageMeter('U', ':.4e')
    losses_in = AverageMeter('In', ':.4e')
    progress = ProgressMeter(
        n_iters_per_epoch,
        [batch_time, data_time, losses_x, losses_u, losses_in],
        prefix="Epoch: [{}]".format(epoch)
    )
    end = time.time()

    # dltrain_x.sampler.set_epoch(epoch)
    # dltrain_u.sampler.set_epoch(epoch)
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    ulbl_loss_dict = dict()
    model.train()
    for i in range(n_iters_per_epoch):

        data_time.update(time.time() - end)

        ims_x_weak, lbs_x, index_x = next(dl_x)
        (ims_u_weak, ims_u_strong), lbs_u_real, index_u = next(dl_u)

        lbs_x = lbs_x.cuda(local_rank, non_blocking=True)
        index_x = index_x.cuda(local_rank, non_blocking=True)
        # index_u = index_u.cuda(local_rank, non_blocking=True)
        lbs_u_real = lbs_u_real.cuda(local_rank, non_blocking=True)
        ims_x_weak = ims_x_weak.cuda(local_rank, non_blocking=True)
        ims_u_weak = ims_u_weak.cuda(local_rank, non_blocking=True)
        ims_u_strong = ims_u_strong.cuda(local_rank, non_blocking=True)

        logits_x, pseudo_label, logits_u_s, loss_in = model(
                                                        ims_x_weak, ims_u_weak, ims_u_strong, 
                                                        labels=lbs_x, index=index_x, start_unlabel=epoch>0, args=args
                                                    )
        loss_x = F.cross_entropy(logits_x, lbs_x, reduction='mean')
        
        
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        loss_u = (torch.sum(-F.log_softmax(logits_u_s,dim=1) * pseudo_label.detach(), dim=1) * mask).mean()
        
        if epoch == args.epochs-1:
            if index_u not in ulbl_loss_dict.keys():
                ulbl_loss_dict[index_u] = loss_u
            if index_u in ulbl_loss_dict.keys():
                ulbl_loss_dict[index_u] += loss_u
        
        loss_in = loss_in.mean()
        loss = loss_x + loss_u + loss_in * args.lambda_in

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses_x.update(loss_x.item())
        losses_u.update(loss_u.item())
        losses_in.update(loss_in.item())

        if rank == 0 and i % 10 == 0:
            progress.display(i)
            
    if epoch == args.epochs-1:
        return [(ulbl_loss_dict[key], key) for key in ulbl_loss_dict.keys()]
        

@torch.no_grad()
def test(model,  test_loader, local_rank):
    model.eval()
    # ---------------------- Test --------------------------
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    ema_top1 = AverageMeter('EMA@1', ':6.2f')
    ema_top5 = AverageMeter('EMA@5', ':6.2f')
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            
            out = model.encoder_q(image)
            acc1, acc5 = accuracy(out, label, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            ema_out = model.ema(image)
            ema_acc1, ema_acc5 = accuracy(ema_out, label, topk=(1, 5))
            ema_top1.update(ema_acc1[0], image.size(0))
            ema_top5.update(ema_acc5[0], image.size(0))


    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, ema_top1.sum, ema_top1.count, ema_top5.sum, ema_top5.count)
    ema_top1_acc = sum(sum1.float()) / sum(cnt1.float())
    ema_top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc, ema_top1_acc, ema_top5_acc

def main(dltrain_x, dltrain_u, test_dataset, num_classes, weight_decay, base_model, episode, init_state_dict=None):
    rank, local_rank, world_size = 0,0,1
    
    batch_size = 64 // world_size
    n_iters_per_epoch = 1024
    lr = 0.03
    mu = 7

    model = SimMatch(base_encoder=base_model, num_classes=num_classes, K=len(dltrain_x.dataset), args=args)
    model.cuda()

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.SGD(grouped_parameters, lr=lr, momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs*n_iters_per_epoch)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    best_acc1 = best_acc5 = 0
    best_ema1 = best_ema5 = 0


    if not os.path.exists('checkpoints') and rank==0:
        os.makedirs('checkpoints')

    #조금 더 고민 필요
    checkpoint_path = f'./checkpoints/seed{args.seed}_epi{episode}_{args.checkpoint}'
    load_path = f'./checkpoints/seed{args.seed}_epi{episode-1}_{args.checkpoint}'
    print('checkpoint_path:', checkpoint_path)
    if not(init_state_dict==None):
        model.load_state_dict(init_state_dict, strict=False)
    if os.path.exists(load_path) and args.load:
        checkpoint = torch.load(load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        if epoch == epochs-1:
            ulbl_loss_list = train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, local_rank, rank)
        else:
            train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, local_rank, rank)
        top1_acc, top5_acc, ema_top1_acc, ema_top5_acc = test(model, test_loader, local_rank)

        best_acc1 = max(top1_acc, best_acc1)
        best_acc5 = max(top5_acc, best_acc5)
        best_ema1 = max(ema_top1_acc, best_ema1)
        best_ema5 = max(ema_top5_acc, best_ema5)

        if rank == 0:
            print('Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc1, best_acc5=best_acc5))
            print('Epoch:{} * EMA@1 {top1_acc:.3f} EMA@5 {top5_acc:.3f} Best_EMA@1 {best_acc:.3f} Best_EMA@5 {best_acc5:.3f}'.format(epoch, top1_acc=ema_top1_acc, top5_acc=ema_top5_acc, best_acc=best_ema1, best_acc5=best_ema5))

            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)
        
        #Sampling Algorithm 추가, unlbl_idx, lbl_idx, model_para를 받아서 unlbl에서 lbl로 보낼 것들을 선별
        #위에서 선별된 결과를 return 시켜 다음 loader를 만들때 반영
        K = 400 if episode==0 else 500
        ulbl_loss_list.sort(key=lambda x:x[0], reverse=True)
        return [data[1] for data in ulbl_loss_list[:K]]

if __name__ == "__main__":
    rank, local_rank, world_size = 0,0,1
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    batch_size = 64 // world_size
    n_iters_per_epoch = 1024
    lr = 0.01
    mu = 7
    
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=get_test_augment('cifar10'))
        num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=get_test_augment('cifar100'))
        num_classes = 100
        
    if args.dataset == 'cifar100':
        weight_decay = 1e-3
        base_model = wide_resnet28w8()
    else:
        weight_decay = 5e-4
        # base_model = wide_resnet28w2()
        base_model = resnet18()
    
    indexes = None
    for episode in range(args.episodes):
        if episode == 0:
            init_data = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/SimCLR-2/runs/Sep19_02-36-34_ailab-server-bengio/checkpoints/sampled_idx.pkl'
            with open(init_data, 'rb') as f:
                init_data_list = pickle.load(f)
            init_data_idx = [data[1] for data in init_data_list]
            ulbl_data_idx = [i for i in range(50000) if i not in init_data_idx]
            indexes = [init_data_idx,ulbl_data_idx]
            init_para = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/SimCLR-2/runs/Sep19_02-36-34_ailab-server-bengio/checkpoints/model.pth'
            init_state_dict = torch.load(init_para)
            init_state_dict = std_convert(init_state_dict)
            
        dltrain_x, dltrain_u, labeled_idx, unlabeled_idx = get_fixmatch_data(
                                                                            dataset=args.dataset, 
                                                                            label_per_class=args.label_per_class, 
                                                                            batch_size=batch_size, 
                                                                            n_iters_per_epoch=n_iters_per_epoch, 
                                                                            mu=mu, dist=False, return_index=True,
                                                                            indexes=indexes
                                                                        )
        np.savetxt(f'./checkpoints/seed{args.seed}_epi{episode}_labeled.txt', labeled_idx)
        np.savetxt(f'./checkpoints/seed{args.seed}_epi_{episode}_unlabeled.txt', unlabeled_idx)
        
        sampled_idx = main(dltrain_x, dltrain_u, test_dataset, num_classes, weight_decay, base_model, episode, init_state_dict)
        #main으로 부터 indexes 관련 받아서
        # indexes = ...
        lbl_data_idx = lbl_data_idx + sampled_idx
        ulbl_data_idx = [idx for idx in ulbl_data_idx if idx not in sampled_idx]
        indexes = [lbl_data_idx, ulbl_data_idx]
        init_state_dict = None

