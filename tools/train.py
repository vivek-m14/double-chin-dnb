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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb
import sys
import datetime

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unet import BaseUNetHalf as UNet
from src.data.dataset import create_data_loaders
from src.losses.losses import CombinedLoss
from src.blend.blend_map import apply_flow_formula_batch
from src.utils.utils import load_checkpoint, save_visualization_batch, compute_metrics


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
    running_losses = {'total_loss': 0.0, 'tensor_map_loss': 0.0,
                    'blend_masked_loss': 0.0, 'blend_unmasked_loss': 0.0,
                    'image_mse_loss': 0.0, 'perc_loss': 0.0, 'tv_loss': 0.0}
    
    if local_rank == 0:
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                        desc=f"Training Epoch [{epoch + 1}/{num_epochs}]")
    else:
        train_bar = enumerate(train_loader)

    for batchc_idx, batch in train_bar:
        # Move tensors to the correct device
        images = batch['image'].to(device)
        target_tensor_maps = batch['tensor_map'].to(device)
        gt_images = batch['gt'].to(device)

        optimizer.zero_grad()

        # Forward pass - predict 3-channel blend map
        pred_tensor_maps = model(images)
        
        # Apply blend formula to get retouched images
        retouched_images = apply_flow_formula_batch(images, pred_tensor_maps)

        # Calculate combined loss
        loss, losses_dict = criterion(pred_tensor_maps, target_tensor_maps, retouched_images, gt_images)

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
    running_losses = {'total_loss': 0.0, 'tensor_map_loss': 0.0,
                    'blend_masked_loss': 0.0, 'blend_unmasked_loss': 0.0,
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
            target_tensor_maps = batch['tensor_map'].to(device)
            gt_images = batch['gt'].to(device)
            
            # Create a new batch dict with device-moved tensors for visualization
            batch_for_vis = {
                'image': images,
                'tensor_map': target_tensor_maps,
                'gt': gt_images,
                'filename': batch['filename']  # Filenames don't need to move to device
            }

            # Forward pass - predict 3-channel blend map
            pred_tensor_maps = model(images)
            
            # Apply blend formula to get retouched images
            retouched_images = apply_flow_formula_batch(images, pred_tensor_maps).to(device)

            # Calculate combined loss
            loss, losses_dict = criterion(pred_tensor_maps, target_tensor_maps, retouched_images, gt_images)

            # Calculate metrics
            metrics = compute_metrics(retouched_images, gt_images)
            
            # Update running losses and metrics
            for key in running_losses:
                running_losses[key] += losses_dict.get(key, 0.0)
                
            for key in running_metrics:
                running_metrics[key] += metrics.get(key, 0.0)

            if local_rank == 0:
                test_bar.set_postfix(Loss=loss.item())

            # Save some test results for visualization
            if local_rank == 0 and save_dir and batch_idx < 10:  # Save first 2 batches
                vis_save_dir = os.path.join(save_dir, 'results', f'epoch_{epoch+1}')
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)
                save_visualization_batch(batch_for_vis, pred_tensor_maps, vis_save_dir, prefix=f"test_{batch_idx}", max_samples=15)

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
    
    # Set environment variables for debugging
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Set up distributed training
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank}: Set device to {device}")
    
    # Initialize wandb for the main process
    if local_rank == 0 and args.get('use_wandb', True):
        wandb.init(
            project=args.get('project_name', 'double-chin-warp-maps'),
            entity=args.get('entity_name', 'aka-vm'),
            config=args
        )
    
    print(f"Process {local_rank}: Creating data loaders")
    # Create data loaders
    train_loader, test_loader, train_sampler, _ = create_data_loaders(
        args, world_size=world_size, rank=local_rank
    )
    print(f"Process {local_rank}: Data loaders created")
    
    print(f"Process {local_rank}: Creating model")
    # Create model
    model = UNet(n_channels=3, n_classes=2)
    model = model.to(device)
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
    
    print(f"Process {local_rank}: Wrapping model with DDP")
    try:
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        print(f"Process {local_rank}: Model wrapped with DDP")
        
        # Ensure all processes are synchronized
        print(f"Process {local_rank}: Waiting at barrier")
        dist.barrier()
        print(f"Process {local_rank}: Passed barrier")
    except Exception as e:
        print(f"Process {local_rank}: Failed to initialize DDP: {e}")
        raise e
    
    # Create loss function
    criterion = CombinedLoss(
        lambda_blend=args.get('lambda_blend', 1.0),
        lambda_blend_masked=args.get('lambda_blend_masked', 1.0),
        lambda_blend_unmasked=args.get('lambda_blend_unmasked', 0.0),
        lambda_image_mse=args.get('lambda_image_mse', 1.0),
        lambda_perc=args.get('lambda_perc', 0.1),
        lambda_tv=args.get('lambda_tv', 0.1),
        blend_mask_threshold=args.get('blend_mask_threshold', 0.02),
    ).to(device)  # Explicitly move criterion to device
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.get('learning_rate', 0.0001))
    scheduler = StepLR(
        optimizer, 
        step_size=args.get('lr_step_size', 50), 
        gamma=args.get('lr_gamma', 0.1)
    )
    
    # Create save directory
    save_dir = args.get('save_dir', 'weights')
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'results')):
            os.makedirs(os.path.join(save_dir, 'results'))
    
    # Training loop
    best_test_loss = float('inf')
    best_epoch = -1
    start_epoch = args.get('start_epoch', 0)
    num_epochs = args.get('num_epochs', 100)
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            local_rank, world_size, epoch, num_epochs
        )
        
        # Log training results
        if local_rank == 0 and args.get('use_wandb', True):
            wandb_log = {
                'train_loss': train_losses['total_loss'],
                'train_tensor_map_loss': train_losses['tensor_map_loss'],
                'train_blend_masked_loss': train_losses['blend_masked_loss'],
                'train_blend_unmasked_loss': train_losses['blend_unmasked_loss'],
                'train_image_mse_loss': train_losses['image_mse_loss'],
                'train_perc_loss': train_losses['perc_loss'],
                'train_tv_loss': train_losses['tv_loss'],
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr']
            }
            wandb.log(wandb_log)
            
            print(f"\nEpoch [{epoch + 1}/{num_epochs}], Training Loss: {train_losses['total_loss']:.4f}")
        
        # Validate
        test_results = validate(
            model, test_loader, criterion, device, local_rank, 
            world_size, epoch, num_epochs, save_dir
        )
        
        # Log validation results
        if local_rank == 0:
            print(f"\nTesting Statistics:")
            print(f"Average Testing Loss: {test_results['total_loss']:.4f}")
            print(f"Average PSNR: {test_results['psnr']:.2f} dB")
            
            if args.get('use_wandb', True):
                wandb_log = {
                    'test_loss': test_results['total_loss'],
                    'test_tensor_map_loss': test_results['tensor_map_loss'],
                    'test_blend_masked_loss': test_results['blend_masked_loss'],
                    'test_blend_unmasked_loss': test_results['blend_unmasked_loss'],
                    'test_image_mse_loss': test_results['image_mse_loss'],
                    'test_perc_loss': test_results['perc_loss'],
                    'test_tv_loss': test_results['tv_loss'],
                    'test_psnr': test_results['psnr'],
                    'epoch': epoch + 1
                }
                wandb.log(wandb_log)
            
            # Save best model
            if test_results['total_loss'] < best_test_loss:
                best_test_loss = test_results['total_loss']
                best_epoch = epoch + 1
                torch.save(model.module.state_dict(), f"{save_dir}/double_chin_warp_bmap_best.pth")
                print(f"New best model saved! (Epoch {best_epoch}, Loss: {best_test_loss:.4f})")
            
            # Save regular checkpoint (every N epochs)
            if (epoch + 1) % args.get('save_interval', 5) == 0:
                torch.save(model.module.state_dict(), f"{save_dir}/double_chin_warp_bmap_epoch_{epoch + 1}.pth")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    if local_rank == 0:
        torch.save(model.module.state_dict(), f'{save_dir}/double_chin_warp_bmap_final.pth')
        print("Finished Training")
        print(f"Best model was from epoch {best_epoch} with test loss: {best_test_loss:.4f}")
        print(f"Final model saved as '{save_dir}/double_chin_warp_bmap_final.pth'")
        
        if args.get('use_wandb', True):
            wandb.finish()


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
            backend='gloo',  # Use gloo backend instead of nccl
            init_method=f'tcp://localhost:{args["port"]}',
            world_size=world_size,
            rank=local_rank,
            timeout=datetime.timedelta(seconds=600)
        )
    except Exception as e:
        print(f"Process {local_rank}: Failed to initialize process group: {e}")
        raise e
    
    # Start training
    train_skin_retouching_model(local_rank, world_size, args)


def load_config(config_path='default.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to configs/default.yaml)
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cleanup():
    """Cleanup function to ensure proper process termination."""
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()


def main():
    # Load configuration from YAML file
    config = load_config()
    
    # Modify save_dir to include project name suffix
    project_suffix = config['project_name'].split("-")[-1]
    config['save_dir'] = config['save_dir'] + project_suffix
    
    # Add distributed training parameters
    world_size = torch.cuda.device_count()
    config['world_size'] = world_size
    config['init_method'] = f'tcp://127.0.0.1:{config["port"]}'
    
    print(f"Starting distributed training with {world_size} GPUs")
    print(f"Configuration loaded from configs/default.yaml")
    print(f"Training parameters: {config['num_epochs']} epochs, batch size {config['batch_size']}, lr {config['learning_rate']}")
    print(f"Save directory: {config['save_dir']}")
    
    try:
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
    