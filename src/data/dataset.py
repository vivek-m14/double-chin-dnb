import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, ColorJitter
import torchvision.transforms.functional as TF
import yaml
from src.blend.blend_map import process_target_blend_map, compute_target_blend_map_np

import random

import json


def get_files_by_base_name(directory, valid_extensions=('.jpg', '.jpeg', '.png')):
    """
    Returns a dictionary mapping base filenames to their full filenames in a directory.
    
    Args:
        directory (str): Directory path to scan
        valid_extensions (tuple): Valid file extensions to include
        
    Returns:
        dict: Dictionary mapping base names to full filenames
    """
    file_dict = {}
    for filename in os.listdir(directory):
        base_name, ext = os.path.splitext(filename)
        if ext.lower() in valid_extensions:
            file_dict[base_name] = filename  # Store full filename
    return file_dict


class BlendMapDataset(Dataset):
    """
    Dataset for skin retouching with original images, blend maps, and ground truth images.
    
    Args:
        image_dir (str): Directory containing original images
        blend_map_dir (str): Directory containing blend maps
        gt_dir (str): Directory containing ground truth (retouched) images
        transform (callable, optional): Optional transform to be applied on a sample
        resize_dim (tuple): Dimensions to resize images to (width, height)
    """
    ORIGINAL_IMAGE = "original_image"
    BLEND_MAP_IMAGE = "blendmap_image"
    GT_IMAGE = "edited_image"
    # ORIGINAL_IMAGE = "original_image_cropped"
    # BLEND_MAP_IMAGE = "blendmap_image_cropped"
    # GT_IMAGE = "edited_image_cropped"

    def __init__(self, data_root, data_json, transform=None, resize_dim=(1024, 1024), test=False):
        self.data_root = data_root
        data_json = data_json

        self.transform = transform
        self.resize_dim = resize_dim

        if test:
            data_map = json.load(open(data_json))[:100]
        else:
            data_map = json.load(open(data_json))

        for data_item in data_map:
            if self.BLEND_MAP_IMAGE not in data_item:
                bmap_path = data_item[self.GT_IMAGE].replace(self.GT_IMAGE, self.BLEND_MAP_IMAGE)
                data_item[self.BLEND_MAP_IMAGE] = os.path.splitext(bmap_path)[0] + '.npy'
        os.makedirs(os.path.join(self.data_root, self.BLEND_MAP_IMAGE), exist_ok=True)
        data_map = [data for data in data_map if data["source"] != "aadan_double_chin"]
        print(f"Length of data_map: {len(data_map)}")
        self.data_map = data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx, retry=0):

        try:
            data_item = self.data_map[idx]

            img_path = data_item[self.ORIGINAL_IMAGE]
            blend_map_path = data_item[self.BLEND_MAP_IMAGE]
            gt_path = data_item[self.GT_IMAGE]

            img_name = img_path.split('/')[-1]
            
            img_path = os.path.join(self.data_root, img_path)
            blend_map_path = os.path.join(self.data_root, blend_map_path)
            gt_path = os.path.join(self.data_root, gt_path)

            # Load images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.resize_dim)

            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, self.resize_dim)

            # Load blend map as RGB (3-channel) instead of grayscale
            if os.path.exists(blend_map_path):
                blend_map = np.load(blend_map_path).astype(np.float32)
                blend_map = cv2.resize(blend_map, self.resize_dim)
            else:
                blend_map = compute_target_blend_map_np(image, gt)
                np.save(blend_map_path, blend_map.astype(np.float16))

            # Normalize
            image = image.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0
            # blend_map = blend_map.astype(np.float32) / 255.0

            # Convert to PyTorch tensors
            if self.transform:
                image = self.transform(image)
                blend_map = self.transform(blend_map)  # Now a 3-channel tensor
                gt = self.transform(gt)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
                blend_map = torch.from_numpy(blend_map).permute(2, 0, 1)  # Now a 3-channel tensor
                gt = torch.from_numpy(gt).permute(2, 0, 1)

            image, blend_map, gt = self.apply_augmentation(image, blend_map, gt)

            return {
                'image': image,
                'blend_map': blend_map,
                'gt': gt,
                'filename': img_name
            }
        except Exception as e:
            if retry > 3:
                raise e
            with open('error_log1.txt', 'a') as f:
                f.write(f"Error in reading {img_path}, {blend_map_path}, {gt_path}: {e}\n")
            print(f"Error in reading {img_path}, {blend_map_path}, {gt_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)), retry=retry+1)

    def apply_augmentation(self, image, blend_map, gt):
        """Apply the same augmentation to all three components"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            blend_map = TF.hflip(blend_map)
            gt = TF.hflip(gt)

        static_angle=0
        if torch.rand(1) < 0.2:
            angles = [90, 180, 45]
            static_angle = angles[torch.randint(0, len(angles), (1,)).item()]

        # Range rotation (e.g., -30 to 30 degrees)
        if torch.rand(1) < 0.5:
            rotation_range = 30 + static_angle
            angle = torch.randint(-rotation_range, rotation_range, (1,)).item()
            image = TF.rotate(image, angle, expand=False)
            gt = TF.rotate(gt, angle, expand=False)
            blend_map = TF.rotate(blend_map, angle, fill=(0.5, 0.5, 0.5), expand=False)

        # # color jitter
        # if torch.rand(1) < 0.3:
        #     color_jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        #     image = color_jitter(image)
        #     gt = color_jitter(gt)
        #     blend_map = color_jitter(blend_map)

        # if torch.rand(1) < 0.1: # Convert to grayscale
        #     image = TF.rgb_to_grayscale(image)
        #     gt = TF.rgb_to_grayscale(gt)
        #     blend_map = TF.rgb_to_grayscale(blend_map)

        return image, blend_map, gt


class TensorMapDataset(Dataset):
    """
    Dataset for tensor map generation.
    
    Args:
        data_root (str): Directory containing original images
        data_json (str): Path to the json file containing the data
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, data_root, data_json, transform=None, resize_dim=(1024, 1024)):
        self.data_root = data_root
        data_json = data_json

        self.tensor_map_type = 'npy'

        self.data_map = json.load(open(data_json))

        self.transform = transform
        self.resize_dim = resize_dim
        
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):

        try:
            data_item = self.data_map[idx]

            img_path = data_item['original_image']
            tensor_map_path = data_item['flow_map']
            gt_path = data_item['edited_image']

            img_name = img_path.split('/')[-2]
            
            img_path = os.path.join(self.data_root, img_path)
            tensor_map_path = os.path.join(self.data_root, tensor_map_path)
            gt_path = os.path.join(self.data_root, gt_path)

            # Load images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.resize_dim)
            image = image.astype(np.float32) / 255.0

            # Load tensor map
            if self.tensor_map_type == 'npy':
                tensor_map = np.load(tensor_map_path)
                tensor_map = process_target_blend_map(tensor_map)
            elif self.tensor_map_type == 'png':
                tensor_map = cv2.imread(tensor_map_path)
                tensor_map = cv2.cvtColor(tensor_map, cv2.COLOR_BGR2RGB)
            tensor_map = cv2.resize(tensor_map, self.resize_dim)
            tensor_map = tensor_map.astype(np.float32)

            # Load ground truth image
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, self.resize_dim)
            gt = gt.astype(np.float32) / 255.0

            # Convert to PyTorch tensors
            if self.transform:
                image = self.transform(image)
                tensor_map = self.transform(tensor_map)  
                gt = self.transform(gt)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
                tensor_map = torch.from_numpy(tensor_map).permute(2, 0, 1)  # Now a 3-channel tensor
                gt = torch.from_numpy(gt).permute(2, 0, 1)

            # image, tensor_map, gt = self.apply_augmentation(image, tensor_map, gt)

            return {
                'image': image,
                'tensor_map': tensor_map,
                'gt': gt,
                'filename': img_name
            }
        except Exception as e:
            with open('error_log1.txt', 'a') as f:
                f.write(f"Error in reading {img_path}, {tensor_map_path}, {gt_path}: {e}\n")
            return self.__getitem__(np.random.randint(0, len(self)))

    def apply_augmentation(self, image, tensor_map, gt):
        """Apply the same augmentation to all three components"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            tensor_map = TF.hflip(tensor_map)
            gt = TF.hflip(gt)

        static_angle = 0
        if torch.rand(1) < 0.2:
            angles = [90, 180, 45]
            static_angle = angles[torch.randint(0, len(angles), (1,)).item()]

        # Range rotation (e.g., -30 to 30 degrees)
        if torch.rand(1) < 0.5:
            rotation_range = 30 + static_angle
            angle = torch.randint(-rotation_range, rotation_range, (1,)).item()
            image = TF.rotate(image, angle, expand=False)
            gt = TF.rotate(gt, angle, expand=False)
            tensor_map = TF.rotate(tensor_map, angle, fill=(0, 0), expand=False)
            
        return image, tensor_map, gt


def create_data_loaders(args, world_size=None, rank=None, dataset_type='blend_map', test=False):
    """
    Create data loaders for training and testing.
    
    Args:
        args (dict): Arguments containing dataset paths and parameters
        world_size (int, optional): Number of processes for distributed training
        rank (int, optional): Process rank for distributed training
        
    Returns:
        tuple: (train_loader, test_loader, train_sampler, test_sampler)
    """
    # Create dataset
    if dataset_type == 'blend_map':
        dataset = BlendMapDataset(
        args['data_root'], 
        args['data_json'], 
        # args['gt_dir'], 
        transform=ToTensor(), 
        resize_dim=(args['img_size'], args['img_size']),
        test=test
    )
    elif dataset_type == 'tensor_map':
        dataset = TensorMapDataset(
            args['data_root'], 
            args['data_json'], 
            transform=ToTensor(), 
            resize_dim=(args['img_size'], args['img_size'])
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    # Split dataset into train and test
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create samplers for distributed training if world_size and rank are provided
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size'], 
        sampler=train_sampler, 
        num_workers=args.get('num_workers', 4),
        shuffle=(train_sampler is None)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args['batch_size'], 
        sampler=test_sampler, 
        num_workers=args.get('num_workers', 4),
        shuffle=False
    )
    
    return train_loader, test_loader, train_sampler, test_sampler

if __name__ == '__main__':
    config = yaml.load(open('blend_map.yaml'), Loader=yaml.FullLoader)
    train_loader, test_loader, train_sampler, test_sampler = create_data_loaders(config, dataset_type='blend_map')
    
    print(len(train_loader))
    for batch in train_loader:
        print(batch)
        break