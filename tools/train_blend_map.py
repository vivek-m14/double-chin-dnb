"""
Distributed training script for the skin retouching model.
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
import mlflow
import sys
import datetime
import time
import random
import numpy as np
import traceback

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unet import BaseUNetHalf, BaseUNetHalfLite
from src.data.dataset import create_data_loaders
from src.losses.losses import CombinedLoss
from src.blend.blend_map import apply_blend_formula
from src.utils.utils_blend import load_checkpoint, save_visualization_batch, compute_metrics, save_full_checkpoint, load_full_checkpoint, get_git_sha, CSVMetricsLogger


def train_epoch(model, train_loader, optimizer, criterion, device, local_rank, world_size, epoch, num_epochs):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on
        local_rank: Local process rank
        world_size: Total number of processes
        epoch: Current epoch number
        num_epochs: Total number of epochs
        
    Returns:
        dict: Dictionary of average losses
    """
    model.train()
    running_losses = {'total_loss': 0.0, 'blend_map_loss': 0.0, 
                    'image_mse_loss': 0.0, 'perc_loss': 0.0, 'tv_loss': 0.0}
    
    if local_rank == 0:
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"Training Epoch [{epoch + 1}/{num_epochs}]")
    else:
        train_bar = enumerate(train_loader)

    for batch_idx, batch in train_bar:
        # Move tensors to the correct device
        images = batch['image'].to(device)
        target_blend_maps = batch['blend_map'].to(device)
        gt_images = batch['gt'].to(device)

        optimizer.zero_grad()

        # Forward pass - predict 3-channel blend map
        pred_blend_maps = model(images)
        
        # Apply blend formula to get retouched images
        retouched_images = apply_blend_formula(images, pred_blend_maps)

        # Calculate combined loss
        loss, losses_dict = criterion(pred_blend_maps, target_blend_maps, retouched_images, gt_images)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update running losses
        for key in running_losses:
            running_losses[key] += losses_dict.get(key, 0.0)

        if local_rank == 0:
            train_bar.set_postfix(Loss=loss.item())

    # Reduce losses across all processes
    for key in running_losses:
        loss_tensor = torch.tensor(running_losses[key]).to(device)
        dist.reduce(loss_tensor, dst=0)
        if local_rank == 0:
            running_losses[key] = loss_tensor.item() / (len(train_loader) * world_size)

    return running_losses


def validate(model, test_loader, criterion, device, local_rank, world_size, epoch, num_epochs, save_dir=None):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to validate on
        local_rank: Local process rank
        world_size: Total number of processes
        epoch: Current epoch number
        num_epochs: Total number of epochs
        save_dir: Directory to save visualizations (optional)
        
    Returns:
        dict: Dictionary of average losses and metrics
    """
    model.eval()
    running_losses = {'total_loss': 0.0, 'blend_map_loss': 0.0, 
                    'image_mse_loss': 0.0, 'perc_loss': 0.0, 'tv_loss': 0.0}
    running_metrics = {'psnr': 0.0}
    
    if local_rank == 0:
        test_bar = tqdm(enumerate(test_loader), total=len(test_loader), 
                       desc=f"Testing Epoch [{epoch + 1}/{num_epochs}]")
    else:
        test_bar = enumerate(test_loader)

    with torch.no_grad():
        for batch_idx, batch in test_bar:
            # Move all batch tensors to the correct device
            images = batch['image'].to(device)
            target_blend_maps = batch['blend_map'].to(device)
            gt_images = batch['gt'].to(device)
            
            # Create a new batch dict with device-moved tensors for visualization
            batch_for_vis = {
                'image': images,
                'blend_map': target_blend_maps,
                'gt': gt_images,
                'filename': batch['filename']  # Filenames don't need to move to device
            }

            # Forward pass - predict 3-channel blend map
            pred_blend_maps = model(images)
            
            # Apply blend formula to get retouched images
            retouched_images = apply_blend_formula(images, pred_blend_maps)

            # Calculate combined loss
            loss, losses_dict = criterion(pred_blend_maps, target_blend_maps, retouched_images, gt_images)

            # Calculate metrics
            metrics = compute_metrics(retouched_images, gt_images)
            
            # Update running losses and metrics
            for key in running_losses:
                running_losses[key] += losses_dict.get(key, 0.0)
                
            for key in running_metrics:
                running_metrics[key] += metrics.get(key, 0.0)

            if local_rank == 0:
                test_bar.set_postfix(Loss=loss.item())

            # Save visualizations for first 10 batches
            if local_rank == 0 and save_dir and batch_idx < 10:
                vis_save_dir = os.path.join(save_dir, 'results', f'epoch_{epoch+1}')
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)
                save_visualization_batch(batch_for_vis, pred_blend_maps, vis_save_dir, prefix=f"test_{batch_idx}", max_samples=15)

    # Reduce losses and metrics across all processes
    for key in running_losses:
        loss_tensor = torch.tensor(running_losses[key]).to(device)
        dist.reduce(loss_tensor, dst=0)
        if local_rank == 0:
            running_losses[key] = loss_tensor.item() / (len(test_loader) * world_size)
            
    for key in running_metrics:
        metric_tensor = torch.tensor(running_metrics[key]).to(device)
        dist.reduce(metric_tensor, dst=0)
        if local_rank == 0:
            running_metrics[key] = metric_tensor.item() / (len(test_loader) * world_size)

    # Combine results
    results = {**running_losses, **running_metrics}
    return results


def train_skin_retouching_model(local_rank, world_size, args):
    """
    Main training function.
    
    Args:
        local_rank: Local process rank
        world_size: Total number of processes
        args: Dictionary of arguments
    """
    print(f"Process {local_rank}: Starting initialization")
    
    # Reproducibility
    seed = args.get('seed', 42)
    random.seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    torch.cuda.manual_seed_all(seed + local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for debugging
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Set up distributed training
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank}: Set device to {device}")
    
    # Initialize MLflow for the main process
    if local_rank == 0 and args.get('use_mlflow', True):
        mlflow.set_tracking_uri(args.get('mlflow_tracking_uri', 'mlruns'))
        mlflow.set_experiment(args['project_name'])
        mlflow.start_run(run_name=args.get('run_name', None))
        mlflow.log_params({
            k: v for k, v in args.items()
            if isinstance(v, (str, int, float, bool))
        })
        mlflow.set_tag('git_sha', args.get('git_sha', 'unknown'))
    
    print(f"Process {local_rank}: Creating data loaders")
    # Create data loaders
    train_loader, test_loader, train_sampler, _ = create_data_loaders(
        args, world_size=world_size, rank=local_rank, test=args.get('test', False)
    )
    print(f"Process {local_rank}: Data loaders created")
    
    print(f"Process {local_rank}: Creating model")
    # Create model
    ModelClass = BaseUNetHalfLite if args.get('model_variant', 'default') == 'lite' else BaseUNetHalf
    model = ModelClass(
        n_channels=3, n_classes=3,
        last_layer_activation=args.get('last_layer_activation', 'sigmoid'),
        blend_scale=args.get('blend_scale', 0.5),
    )
    model = model.to(device)
    model_config = {
        'model_variant': args.get('model_variant', 'default'),
        'last_layer_activation': args.get('last_layer_activation', 'sigmoid'),
        'blend_scale': args.get('blend_scale', 0.5),
    }
    print(f"Process {local_rank}: Model created and moved to device")

    # get number of parameters and size in MB
    num_params = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"Process {local_rank}: Number of parameters: {num_params}")
        print(f"Process {local_rank}: Model size: {num_params * 4 / 1024 / 1024:.2f} MB")
    
    # Load pretrained model if available
    if args.get('pretrained_path'):
        if local_rank == 0:
            print(f"Loading pretrained model from {args['pretrained_path']}")
        model = load_checkpoint(model, args['pretrained_path'])
    
    # Ensure VGG16 weights are cached before all ranks try to load them
    # (prevents download race conditions)
    if local_rank == 0:
        from torchvision.models import vgg16, VGG16_Weights
        _ = vgg16(weights=VGG16_Weights.DEFAULT)
        del _
    dist.barrier()  # other ranks wait for rank 0 to finish downloading

    # Create loss function
    criterion = CombinedLoss(
        lambda_blend_mse=args.get('lambda_blend_mse', 1.0),
        lambda_image_mse=args.get('lambda_image_mse', 1.0),
        lambda_perc=args.get('lambda_perc', 0.1),
        lambda_tv=args.get('lambda_tv', 0.1)
    ).to(device)  # Explicitly move criterion to device
    
    # Create optimizer and scheduler (on raw model, before DDP)
    optimizer = optim.Adam(model.parameters(), lr=args.get('learning_rate', 0.0001))
    scheduler_type = args.get('lr_scheduler', 'step')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.get('num_epochs', 100),
            eta_min=args.get('lr_min', 1e-6),
        )
    else:
        scheduler = StepLR(
            optimizer, 
            step_size=args.get('lr_step_size', 50), 
            gamma=args.get('lr_gamma', 0.1)
        )
    
    # Resume from full checkpoint BEFORE DDP wrapping
    best_test_loss = float('inf')
    best_epoch = -1
    start_epoch = args.get('start_epoch', 0)
    resume_path = args.get('resume_path', '')
    if resume_path and os.path.isfile(resume_path):
        ckpt = load_full_checkpoint(resume_path, model, optimizer, scheduler, device=str(device))
        model = ckpt['model'].to(device)
        start_epoch = ckpt['start_epoch']
        best_test_loss = ckpt['best_val_loss']
        best_epoch = start_epoch  # approximate — the true best epoch isn't stored
        if local_rank == 0:
            print(f"Resuming from epoch {start_epoch}, best_val_loss={best_test_loss:.4f}")

    # Flush any pending CUDA errors and sync all ranks before DDP
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"Process {local_rank}: CUDA error before DDP: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise

    print(f"Process {local_rank}: Pre-DDP barrier", flush=True)
    dist.barrier()  # ensure all ranks are alive before DDP
    print(f"Process {local_rank}: Wrapping model with DDP", flush=True)
    try:
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank])
        print(f"Process {local_rank}: Model wrapped with DDP", flush=True)
        
        # Ensure all processes are synchronized
        print(f"Process {local_rank}: Waiting at post-DDP barrier", flush=True)
        dist.barrier()
        print(f"Process {local_rank}: Passed barrier", flush=True)
    except Exception as e:
        print(f"Process {local_rank}: Failed to initialize DDP: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    # Create save directory
    save_dir = args.get('save_dir', 'weights')
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'results')):
            os.makedirs(os.path.join(save_dir, 'results'))
        # Save the config used for this run
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(args, f, default_flow_style=False, sort_keys=False)
    
    # CSV metrics logger (rank 0 only)
    csv_logger = CSVMetricsLogger(save_dir) if local_rank == 0 else None

    # Training loop
    num_epochs = args.get('num_epochs', 100)
    save_interval = args.get('save_interval', 5)
    
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for samplers
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        
        # Only save visualizations every save_interval epochs (and first + last)
        is_vis_epoch = (epoch == start_epoch) or ((epoch + 1) % save_interval == 0) or (epoch + 1 == num_epochs)
        vis_dir = save_dir if is_vis_epoch else None

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            local_rank, world_size, epoch, num_epochs
        )
        
        # Log training results
        if local_rank == 0:
            print(f"\nEpoch [{epoch + 1}/{num_epochs}], Training Loss: {train_losses['total_loss']:.4f}")
            if args.get('use_mlflow', True):
                mlflow.log_metrics({
                    'train_loss': train_losses['total_loss'],
                    'train_blend_map_loss': train_losses['blend_map_loss'],
                    'train_image_mse_loss': train_losses['image_mse_loss'],
                    'train_perc_loss': train_losses['perc_loss'],
                    'train_tv_loss': train_losses['tv_loss'],
                    'lr': optimizer.param_groups[0]['lr'],
                }, step=epoch + 1)
        
        # Validate
        test_results = validate(
            model, test_loader, criterion, device, local_rank, 
            world_size, epoch, num_epochs, vis_dir
        )
        
        # Log validation results
        if local_rank == 0:
            print(f"\nTesting Statistics:")
            print(f"Average Testing Loss: {test_results['total_loss']:.4f}")
            print(f"Average PSNR: {test_results['psnr']:.2f} dB")
            
            if args.get('use_mlflow', True):
                mlflow.log_metrics({
                    'test_loss': test_results['total_loss'],
                    'test_blend_map_loss': test_results['blend_map_loss'],
                    'test_image_mse_loss': test_results['image_mse_loss'],
                    'test_perc_loss': test_results['perc_loss'],
                    'test_tv_loss': test_results['tv_loss'],
                    'test_psnr': test_results['psnr'],
                }, step=epoch + 1)
            
            # CSV logging (always, even if MLflow is off)
            csv_logger.log_epoch(
                epoch=epoch + 1,
                train_losses=train_losses,
                val_results=test_results,
                lr=optimizer.param_groups[0]['lr'],
                epoch_time=time.time() - epoch_start_time,
            )

            # Save best model
            if test_results['total_loss'] < best_test_loss:
                best_test_loss = test_results['total_loss']
                best_epoch = epoch + 1
                save_full_checkpoint(
                    f"{save_dir}/checkpoint_best.pth",
                    model, optimizer, scheduler, epoch, best_test_loss, is_ddp=True,
                    model_config=model_config,
                )
                print(f"New best model saved! (Epoch {best_epoch}, Loss: {best_test_loss:.4f})")
            
            # Rolling latest checkpoint (overwritten every epoch for crash recovery)
            save_full_checkpoint(
                f"{save_dir}/checkpoint_latest.pth",
                model, optimizer, scheduler, epoch, best_test_loss, is_ddp=True,
                model_config=model_config,
            )
        
        # Update learning rate
        scheduler.step()

        epoch_end_time = time.time()
        if local_rank == 0:
            print(f"Epoch {epoch + 1} took {datetime.timedelta(seconds=epoch_end_time - epoch_start_time)}")
            ETA = (num_epochs - epoch - 1) * (epoch_end_time - epoch_start_time)
            print(f"ETA: {datetime.timedelta(seconds=ETA)}")
    
    # Cleanup and final logging
    if local_rank == 0:
        if csv_logger:
            csv_logger.close()
        print("Finished Training")
        print(f"Best model was from epoch {best_epoch} with test loss: {best_test_loss:.4f}")
        
        if args.get('use_mlflow', True):
            mlflow.log_artifact(f'{save_dir}/checkpoint_best.pth')
            mlflow.end_run()


def main_worker(local_rank, world_size, args):
    """
    Worker function for distributed training.
    
    Args:
        local_rank: Local process rank
        world_size: Total number of processes
        args: Dictionary of arguments
    """
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args['port'])
    
    # Initialize process group with timeout
    try:
        dist.init_process_group(
            backend='nccl',  # Use nccl for multi-GPU CUDA training
            init_method=f'tcp://localhost:{args["port"]}',
            world_size=world_size,
            rank=local_rank,
            timeout=datetime.timedelta(seconds=600)
        )
    except Exception as e:
        print(f"Process {local_rank}: Failed to initialize process group: {e}")
        raise e
    
    # Start training (with proper NCCL cleanup on crash)
    try:
        train_skin_retouching_model(local_rank, world_size, args)
    except Exception as e:
        print(f"Process {local_rank}: Training crashed with: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


def load_config(config_path='blend_map.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to configs/default.yaml)
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config.get('test', False):    
        config['use_mlflow'] = False
        config['test'] = True
        config['project_name'] = config['project_name'] + '_test'
        config['save_dir'] = config['save_dir'].rstrip('/') + '_test'

    return config


def cleanup():
    """Cleanup function to ensure proper process termination."""
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed blend-map training")
    parser.add_argument("--config", "-c", default="blend_map.yaml", help="YAML config path")
    cli = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(cli.config)
    
    # Modify save_dir to include project name suffix and timestamp
    project_suffix = config['project_name'].split("-")[-1]
    base_save_dir = os.path.join(config['save_dir'], project_suffix)
    if not config.get('resume_path'):
        run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config['save_dir'] = os.path.join(base_save_dir, run_tag)
    else:
        config['save_dir'] = base_save_dir
    
    # Add distributed training parameters
    world_size = torch.cuda.device_count()
    config['world_size'] = world_size
    config['init_method'] = f'tcp://127.0.0.1:{config["port"]}'
    config['git_sha'] = get_git_sha()
    
    print(f"Starting distributed training with {world_size} GPUs")
    print(f"Git SHA: {config['git_sha']}")
    print(f"Training parameters: {config['num_epochs']} epochs, batch size {config['batch_size']}, lr {config['learning_rate']}")
    print(f"Save directory: {config['save_dir']}")
    
    try:
        mp.set_start_method('spawn', force=True)
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, config))
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        cleanup()
    except Exception as e:
        print(f"Training failed with error: {e}")
        cleanup()
    