"""TrackNet Training Script

Usage:
    python train.py --config config.yaml
"""

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import numpy as np

from model.loss import WeightedBinaryCrossEntropy
from model.tracknet_v4 import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset
from utils.metrics import TrackNetMetrics


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)['train']


def setup_ddp():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, cfg, rank=0, local_rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float('inf')
        self.device = self._get_device()
        self.step = 0
        self.checkpoint = None
        self._setup_dirs()
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.world_size > 1:
            return torch.device(f'cuda:{self.local_rank}')
        if self.cfg['device'] == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        return torch.device(self.cfg['device'])

    def _setup_dirs(self):
        resume_dir = self.cfg.get('resume')
        if resume_dir:
            self.save_dir = Path(resume_dir)
            if not self.save_dir.exists():
                raise FileNotFoundError(f"Resume directory not found: {self.save_dir}")
            checkpoint_dir = self.save_dir / "checkpoints"
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
            if not checkpoint_files:
                checkpoint_files = sorted(checkpoint_dir.glob("emergency_*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
            latest_checkpoint = checkpoint_files[-1]
            self.checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            self.start_epoch = self.checkpoint['epoch'] + (0 if self.checkpoint.get('is_emergency', False) else 1)
            self.step = self.checkpoint.get('step', 0)
            self.best_loss = self.checkpoint.get('best_loss', self.checkpoint.get('val_loss', float('inf')))
            if self.is_main:
                # Resume W&B run with saved run_id if available
                wandb_run_id = self.checkpoint.get('wandb_run_id')
                wandb.init(
                    project=self.cfg.get('wandb_project', 'tracknet'),
                    entity=self.cfg.get('wandb_entity'),
                    name=self.save_dir.name,
                    config=self.cfg,
                    id=wandb_run_id,
                    resume='must' if wandb_run_id else 'allow',
                    dir=str(self.save_dir)
                )
                print(f"Resuming from epoch {self.start_epoch}, step {self.step}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(self.cfg['out']) / f"{self.cfg['name']}_{timestamp}"
            if self.is_main:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / "checkpoints").mkdir(exist_ok=True)
                wandb.init(
                    project=self.cfg.get('wandb_project', 'tracknet'),
                    entity=self.cfg.get('wandb_entity'),
                    name=self.save_dir.name,
                    config=self.cfg,
                    dir=str(self.save_dir)
                )
        if self.world_size > 1:
            dist.barrier()

    def _interrupt(self, signum, frame):
        self.interrupted = True

    def _get_lr(self):
        if self.cfg['lr'] is not None:
            return self.cfg['lr']
        defaults = {'Adadelta': 1.0, 'Adam': 0.001, 'AdamW': 0.001, 'SGD': 0.01}
        return defaults[self.cfg['optimizer']]

    def _calculate_effective_lr(self):
        if self.cfg['optimizer'] == 'Adadelta':
            if not hasattr(self.optimizer, 'state') or not self.optimizer.state:
                return self._get_lr()
            effective_lrs = []
            eps = self.optimizer.param_groups[0].get('eps', 1e-6)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        continue
                    square_avg = state.get('square_avg')
                    acc_delta = state.get('acc_delta')
                    if square_avg is not None and acc_delta is not None:
                        if torch.is_tensor(square_avg) and torch.is_tensor(acc_delta):
                            rms_delta = (acc_delta + eps).sqrt().mean()
                            rms_grad = (square_avg + eps).sqrt().mean()
                            if rms_grad > eps:
                                effective_lr = self._get_lr() * rms_delta / rms_grad
                                effective_lrs.append(effective_lr.item())
            if effective_lrs:
                return max(sum(effective_lrs) / len(effective_lrs), eps)
            return self._get_lr()
        return self.optimizer.param_groups[0]['lr']

    def setup_data(self):
        dataset = FrameHeatmapDataset(self.cfg['data'])
        torch.manual_seed(self.cfg['seed'])
        train_size = int(self.cfg['split'] * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(train_ds, shuffle=True)
            self.val_sampler = DistributedSampler(val_ds, shuffle=False)
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg['batch'], sampler=self.train_sampler,
                num_workers=self.cfg['workers'], pin_memory=True
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.cfg['batch'], sampler=self.val_sampler,
                num_workers=self.cfg['workers'], pin_memory=True
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg['batch'], shuffle=True,
                num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda'
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.cfg['batch'], shuffle=False,
                num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda'
            )

        if self.is_main and not self.checkpoint:
            wandb.config.update({'train_size': len(train_ds), 'val_size': len(val_ds)})

            # Log dataset artifact
            if self.cfg.get('log_artifacts', True):
                dataset_artifact = wandb.Artifact(
                    name="padel-dataset",
                    type="dataset",
                    description=f"TrackNet dataset with {len(dataset)} samples"
                )
                dataset_artifact.add_reference(f"file://{Path(self.cfg['data']).resolve()}")
                wandb.log_artifact(dataset_artifact)

    def _create_optimizer(self):
        lr = self._get_lr()
        wd = self.cfg['wd']
        optimizers = {
            'Adadelta': lambda: torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=wd),
            'Adam': lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd),
            'AdamW': lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd),
            'SGD': lambda: torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        }
        return optimizers[self.cfg['optimizer']]()

    def setup_model(self):
        dropout = self.cfg.get('dropout', 0.0)
        self.model = TrackNet(dropout=dropout).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()
        if self.cfg['scheduler'] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.cfg['factor'],
                patience=self.cfg['patience'], min_lr=self.cfg['min_lr']
            )
        else:
            self.scheduler = None
        if self.checkpoint:
            state_dict = self.checkpoint['model_state_dict']
            if self.world_size > 1 and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif self.world_size == 1 and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            if 'optimizer_state_dict' in self.checkpoint:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

        # Watch model gradients and parameters
        if self.is_main and self.cfg.get('watch_model', True):
            watch_freq = self.cfg.get('watch_freq', 100)
            wandb.watch(self.model, self.criterion, log="all", log_freq=watch_freq)

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        if not self.is_main:
            return None, False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_emergency': is_emergency,
            'step': self.step,
            'timestamp': timestamp,
            'wandb_run_id': wandb.run.id if wandb.run else None
        }
        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)
        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint['best_loss'] = self.best_loss
            best_model_path = self.save_dir / "checkpoints" / "best_model.pth"
            torch.save(checkpoint, best_model_path)

            # Upload best model to W&B
            if self.cfg.get('save_best_model', True):
                artifact = wandb.Artifact(
                    name=f"best-model-{self.save_dir.name}",
                    type="model",
                    description=f"Best model at epoch {epoch+1} with val_loss {val_loss:.6f}"
                )
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)

            return filepath, True
        return filepath, False

    def validate(self, epoch=None):
        self.model.eval()
        total_loss = 0.0
        metrics = TrackNetMetrics(
            threshold=self.cfg.get('threshold', 0.5),
            tolerance=self.cfg.get('tolerance', 4)
        )
        sample_images = []
        log_images = self.cfg.get('log_images', True)
        images_count = self.cfg.get('log_images_count', 8)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_main)):
                if self.interrupted:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Calculate metrics
                if self.cfg.get('compute_metrics', True):
                    metrics.update(outputs, targets)

                # Collect sample images for logging
                if log_images and len(sample_images) < images_count and self.is_main:
                    sample_images.append((inputs[0].cpu(), outputs[0].cpu(), targets[0].cpu()))

        avg_loss = total_loss / len(self.val_loader)
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        # Log metrics and images
        if self.is_main and self.cfg.get('compute_metrics', True):
            computed_metrics = metrics.compute()
            wandb.log({
                'val/accuracy': computed_metrics['accuracy'],
                'val/precision': computed_metrics['precision'],
                'val/recall': computed_metrics['recall'],
                'val/f1_score': computed_metrics['f1_score'],
                'val/detection_rate': computed_metrics['detection_rate'],
                'val/tp': computed_metrics['tp'],
                'val/tn': computed_metrics['tn'],
                'val/fp': computed_metrics['fp1'] + computed_metrics['fp2'],
                'val/fn': computed_metrics['fn']
            }, step=self.step)

        # Log sample images
        if self.is_main and log_images and sample_images:
            self._log_sample_predictions(sample_images, epoch)

        return avg_loss

    def _log_sample_predictions(self, samples, epoch=None):
        """Log sample predictions to W&B."""
        images = []
        for idx, (input_tensor, pred_tensor, gt_tensor) in enumerate(samples):
            # Extract center frame (index 1) from the 3-frame input
            # Input is [9, H, W] = 3 RGB frames concatenated
            frame = input_tensor[3:6].permute(1, 2, 0).numpy()  # Center frame RGB
            frame = (frame * 255).astype(np.uint8)

            # Prediction and GT heatmaps (center frame)
            pred_heatmap = pred_tensor[1].numpy()
            gt_heatmap = gt_tensor[1].numpy()

            # Create overlay
            pred_overlay = self._create_overlay(frame, pred_heatmap, color=(255, 0, 0))  # Red
            gt_overlay = self._create_overlay(frame, gt_heatmap, color=(0, 255, 0))  # Green

            images.append(wandb.Image(frame, caption=f"Sample {idx+1}: Input"))
            images.append(wandb.Image(pred_overlay, caption=f"Sample {idx+1}: Prediction"))
            images.append(wandb.Image(gt_overlay, caption=f"Sample {idx+1}: Ground Truth"))

        caption = f"Epoch {epoch+1}" if epoch is not None else "Validation"
        wandb.log({f"predictions/{caption}": images}, step=self.step)

    def _create_overlay(self, frame, heatmap, color=(255, 0, 0), alpha=0.5):
        """Create overlay of heatmap on frame."""
        overlay = frame.copy()
        heatmap_normalized = (heatmap * 255).astype(np.uint8)

        # Create colored heatmap
        colored_heatmap = np.zeros_like(overlay)
        for i, c in enumerate(color):
            colored_heatmap[:, :, i] = (heatmap_normalized * c / 255).astype(np.uint8)

        # Blend
        mask = heatmap > 0.1
        overlay[mask] = (overlay[mask] * (1 - alpha) + colored_heatmap[mask] * alpha).astype(np.uint8)
        return overlay

    def train(self):
        self.setup_data()
        self.setup_model()

        if self.is_main:
            print(f"W&B: {wandb.run.url}")
            if self.world_size > 1:
                print(f"Training with {self.world_size} GPUs")

        for epoch in range(self.start_epoch, self.cfg['epochs']):
            if self.interrupted:
                break
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg['epochs']}",
                        leave=False, disable=not self.is_main)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if self.interrupted:
                    pbar.close()
                    val_loss = self.validate(epoch)
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    if self.is_main:
                        if self.cfg.get('alerts', True):
                            wandb.alert(
                                title="Training Interrupted",
                                text=f"Stopped at epoch {epoch+1}. Last val_loss: {val_loss:.6f}",
                                level=wandb.AlertLevel.WARN
                            )
                        wandb.finish()
                    return
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1
                if self.is_main:
                    current_lr = self._calculate_effective_lr()
                    wandb.log({'batch/loss': batch_loss, 'batch/lr': current_lr}, step=self.step)
                pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            pbar.close()
            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate(epoch)
            elapsed = time.time() - start_time
            if self.is_main:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': elapsed
                }, step=self.step)
            if self.scheduler:
                self.scheduler.step(val_loss)
            _, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best and self.is_main and self.cfg.get('alerts', True):
                wandb.alert(
                    title="New Best Model",
                    text=f"Epoch {epoch+1}: val_loss improved to {val_loss:.6f}",
                    level=wandb.AlertLevel.INFO
                )
            if self.world_size > 1:
                dist.barrier()

        # Training complete alert
        if self.is_main:
            if self.cfg.get('alerts', True):
                wandb.alert(
                    title="Training Complete",
                    text=f"Finished {self.cfg['epochs']} epochs. Best val_loss: {self.best_loss:.6f}",
                    level=wandb.AlertLevel.INFO
                )
            wandb.finish()


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    train_cfg = full_cfg['train']
    wandb_cfg = full_cfg.get('wandb', {})
    train_cfg['wandb_project'] = wandb_cfg.get('project', 'tracknet')
    train_cfg['wandb_entity'] = wandb_cfg.get('entity')
    # W&B options
    train_cfg['log_images'] = wandb_cfg.get('log_images', True)
    train_cfg['log_images_count'] = wandb_cfg.get('log_images_count', 8)
    train_cfg['save_best_model'] = wandb_cfg.get('save_best_model', True)
    train_cfg['watch_model'] = wandb_cfg.get('watch_model', True)
    train_cfg['watch_freq'] = wandb_cfg.get('watch_freq', 100)
    train_cfg['alerts'] = wandb_cfg.get('alerts', True)
    train_cfg['compute_metrics'] = wandb_cfg.get('compute_metrics', True)
    train_cfg['log_artifacts'] = wandb_cfg.get('log_artifacts', True)
    gpus = train_cfg.get('gpus', 1)

    if gpus > 1 and 'RANK' not in os.environ:
        cmd = [
            'torchrun',
            f'--nproc_per_node={gpus}',
            sys.argv[0],
            '--config', args.config
        ]
        subprocess.run(cmd)
    else:
        rank, local_rank, world_size = setup_ddp()
        try:
            trainer = Trainer(train_cfg, rank, local_rank, world_size)
            trainer.train()
        finally:
            cleanup_ddp()


if __name__ == "__main__":
    main()
