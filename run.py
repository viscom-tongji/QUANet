import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import os
import random
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any
import json

import util.misc as misc
from util.FSC147 import FSC147
from models import quanet as QUANet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from util.constant import SCALE_FACTOR



def get_args_parser():
    parser = argparse.ArgumentParser('QUANet', add_help=False)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="train or test")
    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--decoder_depth', default=4, type=int, help='Number of FIM layers')
    parser.add_argument('--decoder_head', default=8, type=int, help='Number of attention heads for FIM')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ckpt', default=None, type=str,
                        help='path of resume from checkpoint')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # log related
    parser.add_argument('--val_freq', default=1, type=int, help='check validation every val_freq epochs')

    # encoder parameters
    parser.add_argument('--unfreeze_bert', default=['out.'], type=str, help='whether to unfreeze the BERT encoder layers')
    parser.add_argument('--unfreeze_dino', default=['blocks.6','blocks.7','blocks.8','blocks.9','blocks.10','blocks.11','norm.weight','norm.bias'], type=str, help='whether to unfreeze the dino encoder')

    # decoder choice
    parser.add_argument('--decoder_arch', default='adapter', type=str, choices=['cnn','vit','adapter','moe'], help='decoder architecture')

    # number negative prompt
    parser.add_argument('--use_rank', default=True, type=misc.str2bool, help='whether to perform number rank loss.')
    parser.add_argument('--rank_weight', default=0.1, type=float, help='排序学习强化数目认知的损失系数.')
    parser.add_argument('--rank_choose', default='x_cls', type=str, choices=['cnn_token','x_cls','patch_token'], help='用于与rank prompt对齐的token从编码的还是解码端得到')
    parser.add_argument('--interval_boundery', default=[0,10,20,50,100,200,500,1000,5000], nargs='+', type=int, help='间隔边界')
    parser.add_argument('--interval_value', default=[1,2,3,5,10,20,35,35], nargs='+', type=int, help='间隔值')
    parser.add_argument('--rank_number', default=7, type=int, help='生成数目prompt的个数')

    # 一致性loss
    parser.add_argument('--spilt_weight', default=0.1, type=float, help='权重')

    return parser


class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.save_hyperparameters(args)
        model_args = {
            'fim_depth': self.args.decoder_depth,
            'fim_num_heads': self.args.decoder_head,
            'unfreeze_bert': self.args.unfreeze_bert,
            'unfreeze_dino': self.args.unfreeze_dino,
            'decoder_arch': self.args.decoder_arch,
        }
        self.model = QUANet.QUANet(**model_args)
        self.loss = F.mse_loss
        self.all_loss = self.args.decoder_arch in ('adapter', 'moe')
        self.interval_dict = {}
        self.rank_dict = {}
        self.dict_init()
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        samples, gt_density, boxes, m_flag, prompt_gt, prompt_add = batch
        output, extra_out = self.model(samples, prompt_gt, return_extra=True, coop_require_grad=True)

        # Compute loss function
        mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
        masks = np.tile(mask, (output.shape[0], 1))
        masks = masks.reshape(output.shape[0], 384, 384)
        masks = torch.from_numpy(masks).to(self.device)
        loss = self.loss(output, gt_density)
        loss = (loss * masks / (384 * 384)).sum() / output.shape[0]

        if self.all_loss:
            cnn_x = extra_out['cnn_x']
            vit_x = extra_out['vit_x']
            loss_cnn = self.loss(cnn_x, gt_density)
            loss_vit = self.loss(vit_x, gt_density)
            loss_cnn = (loss_cnn * masks / (384 * 384)).sum() / output.shape[0]
            loss_vit = (loss_vit * masks / (384 * 384)).sum() / output.shape[0]
            loss = loss + 0.1 * loss_cnn + 0.1 * loss_vit

            # 一致性loss 超参数
            drop_rate = 0.8
            interval = 5
            kernel_rate = 1

            # 一致性loss
            def fold(x):
                temp = F.unfold(x.unsqueeze(1), kernel_size=kernel_rate * 16, stride=kernel_rate * 16)
                return temp.sum(dim=1).squeeze(1)

            gt_patch = fold(gt_density)
            cnn_patch = fold(cnn_x)
            vit_patch = fold(vit_x)
            mixed_patch1 = mixed_patch2 = torch.zeros_like(cnn_patch)

            mixed_patch1[:, 0::2] = vit_patch[:, 0::2]
            mixed_patch1[:, 1::2] = cnn_patch[:, 1::2]
            mixed_patch2[:, 0::2] = cnn_patch[:, 0::2]
            mixed_patch2[:, 1::2] = vit_patch[:, 1::2]

            _, indices = torch.sort(gt_patch, dim=1)
            _, n = gt_patch.shape
            mixed_patch1 = torch.gather(mixed_patch1, 1, indices)[:, int(n * drop_rate):]
            mixed_patch2 = torch.gather(mixed_patch2, 1, indices)[:, int(n * drop_rate):]
            cnn_patch = torch.gather(cnn_patch, 1, indices)[:, int(n * drop_rate):]
            vit_patch = torch.gather(vit_patch, 1, indices)[:, int(n * drop_rate):]

            mixed_split_loss2, mixed_split_loss1, cnn_split_loss, vit_split_loss = 0, 0, 0, 0
            for i in range(interval):
                mixed_patch1_i = mixed_patch1[:, i::interval]
                mixed_patch2_i = mixed_patch2[:, i::interval]
                cnn_patch_i = cnn_patch[:, i::interval]
                vit_patch_i = vit_patch[:, i::interval]
                mixed_patch_i_diff1 = mixed_patch1_i[:, :-1] - mixed_patch1_i[:, 1:]
                mixed_patch_i_diff2 = mixed_patch2_i[:, :-1] - mixed_patch2_i[:, 1:]
                cnn_patch_i_diff = cnn_patch_i[:, :-1] - cnn_patch_i[:, 1:]
                vit_patch_i_diff = vit_patch_i[:, :-1] - vit_patch_i[:, 1:]
                mixed_split_loss1 += torch.sum(F.relu(mixed_patch_i_diff1)) / SCALE_FACTOR / SCALE_FACTOR
                mixed_split_loss2 += torch.sum(F.relu(mixed_patch_i_diff2)) / SCALE_FACTOR / SCALE_FACTOR
                cnn_split_loss += torch.sum(F.relu(cnn_patch_i_diff)) / SCALE_FACTOR / SCALE_FACTOR
                vit_split_loss += torch.sum(F.relu(vit_patch_i_diff)) / SCALE_FACTOR / SCALE_FACTOR

            split_loss = (mixed_split_loss1 + mixed_split_loss2 + cnn_split_loss + vit_split_loss) / 2
            split_loss = split_loss / mixed_patch1.shape[0] / mixed_patch1.shape[1]
            loss = loss + self.args.spilt_weight * split_loss

        # number prompt learning
        if self.args.use_rank:
            gt_numbers_np = gt_density.sum(dim=(1, 2)).cpu().numpy()
            gt_numbers_np = np.round(gt_numbers_np / SCALE_FACTOR).astype(int)
            rank_dict = self.rank_prompt(prompt_gt, gt_numbers_np, samples.device)
            index1 = rank_dict['index']
            rank_number_embeddings = rank_dict['embeddings']
            rank_token = extra_out[self.args.rank_choose]

            rank_loss = self.rank_loss(rank_token, rank_number_embeddings, index1)
            loss += self.args.rank_weight * rank_loss
            self.log('train_rank_loss', rank_loss)

        self.log('train_loss', loss)

        # Update information of MAE and RMSE
        batch_mae = 0
        batch_rmse = 0
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()
            gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae += cnt_err
            batch_rmse += cnt_err ** 2
        batch_mae /= output.shape[0]
        batch_rmse /= output.shape[0]
        batch_rmse = math.sqrt(batch_rmse)
        self.log('train_mae', batch_mae)
        self.log('train_rmse', batch_rmse)

        return loss

    def validation_step(self, batch, batch_idx):
        samples, gt_density, _, _, prompt, _, = batch
        output = self.model(samples, prompt)

        # Update information of MAE and RMSE
        batch_mae = []
        batch_rmse = []
        pred_cnts = []
        gt_cnts = []
        for i in range(output.shape[0]):
            pred_cnt = torch.sum(output[i] / SCALE_FACTOR).item()
            gt_cnt = torch.sum(gt_density[i] / SCALE_FACTOR).item()
            cnt_err = abs(pred_cnt - gt_cnt)
            batch_mae.append(cnt_err)
            batch_rmse.append(cnt_err ** 2)
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)

        output = {"mae": batch_mae, "rmse": batch_rmse, "pred_cnts": pred_cnts, "gt_cnts": gt_cnts}
        self.test_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        all_mae = []
        all_rmse = []
        outputs = self.test_outputs
        self.test_outputs = []
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]
        val_mae = np.mean(all_mae)
        val_rmse = np.sqrt(np.mean(all_rmse))
        self.log('val_mae', val_mae)
        self.log('val_rmse', val_rmse)

    def test_step(self, batch, batch_idx):
        image, gt_density, boxes, m_flag, prompt = batch

        assert image.shape[0] == 1, "only support inference one image at a time"
        raw_h, raw_w = image.shape[2:]

        patches, inter = misc.sliding_window(image, stride=128)
        patches = torch.from_numpy(patches).float().to(self.device)
        prompt = np.repeat(prompt, patches.shape[0], axis=0).tolist()
        output, extra = self.model(patches, prompt, return_extra=True)

        output.unsqueeze_(1)
        output = misc.window_composite(output, stride=128)
        output = output.squeeze(1)
        output = output[:, :, :raw_w]

        # Update information of MAE and RMSE
        pred_cnt = torch.sum(output[0] / SCALE_FACTOR).item()
        gt_cnt = torch.sum(gt_density[0] / SCALE_FACTOR).item()
        cnt_err = abs(pred_cnt - gt_cnt)

        result = {"mae": [cnt_err], "rmse": [cnt_err ** 2], "prompt": prompt[0], "pred_cnts": [pred_cnt], "gt_cnts": [gt_cnt]}
        self.test_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        all_mae = []
        all_rmse = []
        outputs = self.test_outputs
        for output in outputs:
            all_mae += output["mae"]
            all_rmse += output["rmse"]

        test_mae = np.mean(all_mae)
        test_rmse = np.sqrt(np.mean(all_rmse))
        self.log('test_mae', test_mae)
        self.log('test_rmse', test_rmse)
        self.test_outputs = []

    def forward(self, img, prompt):
        """
        img: (1, 3, H, W)
        prompt: List[str]
        """
        return self.model(img, prompt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.33)
        return {"optimizer": optimizer, "lr_scheduler": schedular, "monitor": "val_mae"}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        temp_list = list(checkpoint["state_dict"].keys())
        for name, param in self.named_parameters():
            if not param.requires_grad:
                if name in temp_list:
                    del checkpoint["state_dict"][name]
                else:
                    print(f"{name} not in state_dict")

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        self.args = args

    def dict_init(self):
        interval_boundry = self.args.interval_boundery
        interval_val = self.args.interval_value
        for i in range(len(interval_val)):
            for j in range(interval_boundry[i], interval_boundry[i + 1]):
                self.interval_dict[j] = interval_val[i]

    def rank_prompt(self, prompt, gt_numbers_np, device):
        rank_pn = self.args.rank_number
        shape = (len(gt_numbers_np), rank_pn)
        index1 = torch.zeros(shape).bool().to(device)
        weights = torch.zeros(shape).to(device)
        rank_number_prompts = []
        gt_index = []
        for ind in range(len(gt_numbers_np)):
            if gt_numbers_np[ind] in self.rank_dict:
                uiab_tf = self.rank_dict[gt_numbers_np[ind]]['uiab_tf']
                uiab = self.rank_dict[gt_numbers_np[ind]]['uiab']
            else:
                interval = self.interval_dict[gt_numbers_np[ind]]
                rank_np2_s_n = min(np.floor(gt_numbers_np[ind] // interval), round(rank_pn // 2))
                uia = gt_numbers_np[ind] - interval * rank_np2_s_n
                uib = gt_numbers_np[ind] + interval * (rank_pn - 1 - rank_np2_s_n)
                uiab = np.round(np.linspace(uia, uib, rank_pn)).astype(int)
                uiab_tf = np.repeat([True, False], [rank_np2_s_n, rank_pn - rank_np2_s_n])
                self.rank_dict[gt_numbers_np[ind]] = {'uiab': uiab, 'uiab_tf': uiab_tf}
            weights[ind, :] = torch.from_numpy(uiab - gt_numbers_np[ind]).to(weights.device)
            index1[ind, :] = torch.from_numpy(uiab_tf).to(index1.device)
            gt_index.append(uiab_tf.sum())
            uiab_prompts = ["A photo of {} {}".format(num, prompt[ind]) for num in uiab]
            rank_number_prompts.append(uiab_prompts)
        rank_number_prompts = [item for uiab_prompts in rank_number_prompts for item in uiab_prompts]
        rank_number_embeddings = self.model.text_encoder(rank_number_prompts, device).float()
        rank_number_embeddings = rank_number_embeddings.reshape(len(prompt), -1, rank_number_embeddings.shape[1])
        gt_prompts = []
        for i in range(len(prompt)):
            gt_prompts.append(rank_number_embeddings[i, gt_index[i], :])
        gt_prompts = torch.stack(gt_prompts)
        return {'index': index1, 'embeddings': rank_number_embeddings, 'weight': weights, 'gt_prompts': gt_prompts}

    def rank_loss(self, pred_embeddings, rank_embeddings, index):
        rank_token = pred_embeddings
        rank_token = rank_token.unsqueeze(1)
        rank_similarity_matrix = F.cosine_similarity(rank_token, rank_embeddings, dim=2)
        rank_diff_backward = torch.diff(rank_similarity_matrix, dim=1)
        rank_diff_forward = torch.flip(torch.diff(torch.flip(rank_similarity_matrix, dims=[1]), dim=1), dims=[1])

        index_forward = index[:, :-1]
        index_backward = ~index_forward
        rank_diff_forward = rank_diff_forward * index_forward
        rank_diff_backward = rank_diff_backward * index_backward
        rank_diff = rank_diff_forward + rank_diff_backward
        rank_loss = F.relu(rank_diff).sum() / rank_diff.shape[0] / rank_diff.shape[1]
        return rank_loss


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = args.seed
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_train = FSC147(split="train")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    dataset_val = FSC147(split="val")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    save_callback = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=4, mode='min', filename='{epoch}-{val_mae:.2f}')
    model = Model(args)
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.exp_name)
    trainer = Trainer(
        accelerator="gpu",
        callbacks=[save_callback],
        accumulate_grad_batches=args.accum_iter,
        precision=16,
        max_epochs=args.epochs,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
        log_every_n_steps=1,
    )

    if args.mode == "train":
        if args.ckpt is not None:
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
            # Overwrite checkpoint args with current CLI args so new hyperparams take effect
            model.overwrite_args(args)
        trainer.fit(model, train_dataloader, val_dataloader)


    elif args.mode == "test":
        log_d = trainer.logger.log_dir

        dataset_test = FSC147(split="test")

        sampler_test = torch.utils.data.SequentialSampler(dataset_test)


        test_dataloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=1, pin_memory=True, drop_last=False,
        )

        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint to test with --ckpt <path>")

        model = Model.load_from_checkpoint(args.ckpt, strict=False)
        model.overwrite_args(args)
        model.eval()
        model.args.dataset_type = "FSC"

        print("====Metric on test set====")
        test_results = trainer.test(model, test_dataloader)
        print(test_results)

