import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import einops

from steve import STEVE
# from data import GlobVideoDataset
from phyre.dataset import PhyreVideoDataset
from utils import cosine_anneal, linear_warmup

import pickle
import json

def make_batch_padded(list_of_samples):
    max_len = 0
    for video in list_of_samples:
        t = video.shape[0]
        if t > max_len:
            max_len = t
    padded_samples = []
    padding_masks = []
    for video in list_of_samples:
        padding = einops.repeat(video[-1, :, :, :], "c w h -> repeat c w h", repeat = max_len-video.shape[0])
        # padded_video = einops.rearrange([video, padding], "listaxis t c w h -> (listaxis t) c w h")
        padded_video = torch.cat([video, padding], dim=0)
        padded_samples.append(padded_video)
        mask = [i>=video.shape[0] for i in range(max_len)]
        mask = torch.Tensor(mask).bool()
        padding_masks.append(mask)
    # listaxis is batch axis
    padded_samples = einops.rearrange(padded_samples, "listaxis t c w h -> listaxis t c w h")
    padding_masks = einops.rearrange(padding_masks, "listaxis m -> listaxis m")
    return padded_samples, padding_masks

def make_batch_truncated(list_of_samples):
    min_len = 999
    for video in list_of_samples:
        t = video.shape[0]
        if t < min_len:
            min_len = t
    padded_samples = []
    for video in list_of_samples:
        padded_samples.append(video[:min_len])
    # listaxis is batch axis
    padded_samples = einops.rearrange(padded_samples, "listaxis t c w h -> listaxis t c w h")
    return padded_samples, torch.ones_like(padded_samples, dtype=torch.bool)

def collate_fn(list_of_samples):
    samples, padding_mask = make_batch_truncated(list_of_samples)
    return samples


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
# parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
# parser.add_argument('--ep_len', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/*')
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--steps', type=int, default=200000)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=4)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=True, action='store_true')

args = parser.parse_args()
args.image_size=64  # my phyre-dataset kinda hardcodes this

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

# train_dataset = GlobVideoDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
# val_dataset = GlobVideoDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
train_dataset = PhyreVideoDataset(args.data_path)
val_dataset = train_dataset

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=None, collate_fn=collate_fn, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=None, collate_fn=collate_fn, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = 10 # train_epoch_size // 5

model = STEVE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])


def visualize(video, recon_dvae, recon_tf, attns, N=8):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


for epoch in range(start_epoch, args.epochs):
    model.train()
    
    for batch, video in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        video = video.cuda()

        optimizer.zero_grad()
        
        (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)

        if args.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = mse + cross_entropy
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        
        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

#     with torch.no_grad():
                gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
                frames = visualize(video, recon, gen_video, attns, N=8)
                writer.add_video('TRAIN_recons/epoch={:03}/batch={:05}'.format(epoch+1, batch), frames)
    
    with torch.no_grad():
        model.eval()

        val_cross_entropy = 0.
        val_mse = 0.

        for batch, video in enumerate(val_loader):
            video = video.cuda()

            (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()

        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)

        val_loss = val_mse + val_cross_entropy

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if global_step < args.steps:
                torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, f'best_model_until_{args.steps}_steps.pt'))

            if 50 <= epoch:
                gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
                frames = visualize(video, recon, gen_video, attns, N=8)
                writer.add_video('VAL_recons/epoch={:03}'.format(epoch + 1), frames)

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        # save config for two reasons: ease of review, and replication of the same trained architecture over to my side
        with open(os.path.join(log_dir, 'args.pkl'), "wb") as f:
            pickle.dump(args, f)
        with open(os.path.join(log_dir, 'args_readable.json'), "w") as f:
            json.dump(args.vars(), f, sort_keys=True, indent=4)

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
