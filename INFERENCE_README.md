# Double Chin Removal Model - Inference

This directory contains the inference script for the Double Chin Removal model that uses flow-based warping to remove double chins from images.

## Model Overview

The model uses a U-Net architecture to predict flow maps (2-channel displacement fields) that are applied to the input image using grid sampling to achieve double chin removal. The model outputs flow maps in the range [-1, 1] which are then used to warp the original image.

## Files

- `inference.py` - Main inference script with the `DoubleChinInference` class
- `example_inference.py` - Example usage of the inference class
- `INFERENCE_README.md` - This documentation file

## Requirements

Make sure you have the following dependencies installed:
```bash
pip install torch torchvision opencv-python numpy pyyaml
```

## Usage

### Command Line Usage

```bash
python inference.py --input path/to/input/image.jpg --output path/to/output/result.jpg
```

#### Command Line Options

- `--input, -i`: Input image path (required)
- `--output, -o`: Output image path (required)
- `--model, -m`: Model weights path (default: `weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth`)
- `--device, -d`: Device to use (`cpu`, `cuda`, or `auto`) (default: `auto`)
- `--img_size`: Input image size (default: 1024)
- `--save_flow`: Save flow map visualization (optional flag)

#### Examples

```bash
# Basic usage
python inference.py -i input.jpg -o output.jpg

# With custom model path
python inference.py -i input.jpg -o output.jpg -m path/to/model.pth

# Save flow map visualization
python inference.py -i input.jpg -o output.jpg --save_flow

# Use CPU explicitly
python inference.py -i input.jpg -o output.jpg -d cpu
```

### Programmatic Usage

```python
from inference import DoubleChinInference

# Initialize inference
inference = DoubleChinInference(
    model_path="weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth",
    device='auto',
    img_size=1024
)

# Run inference
result = inference.run_inference(
    image_path="input.jpg",
    output_path="output.jpg",
    save_flow_map=True
)
```

### Individual Function Usage

You can also use the individual functions:

```python
from inference import DoubleChinInference

# Initialize
inference = DoubleChinInference(model_path="path/to/model.pth")

# Load image
image = inference.load_image("input.jpg")

# Preprocess
tensor = inference.preprocess(image)

# Run model (you'll need to do this manually)
with torch.no_grad():
    flow_map = inference.model(tensor)

# Postprocess
result = inference.postprocess(flow_map, tensor, image.shape[:2])

# Save
inference.save(result, "output.jpg", save_flow_map=True, flow_map=flow_map)
```

## Model Architecture

The model uses a `BaseUNetHalf` architecture with:
- **Input**: 3-channel RGB image
- **Output**: 2-channel flow map (x, y displacements)
- **Activation**: Tanh activation for flow maps in [-1, 1] range
- **Input size**: 1024x1024 (configurable)

## Flow Map Application

The model predicts flow maps that are applied using the `apply_flow_formula` function, which:
1. Converts flow maps to grid coordinates
2. Uses PyTorch's `grid_sample` for bilinear interpolation
3. Applies the warping to the original image

## Output Files

When running inference, the script generates:
1. **Processed image**: The final result with double chin removal
2. **Flow map visualization** (if `--save_flow` is used): Shows the magnitude of the predicted flow

## Troubleshooting

### Common Issues

1. **Model not found**: Make sure the model weights file exists at the specified path
2. **CUDA out of memory**: Try reducing the `img_size` or using CPU
3. **Import errors**: Make sure the `src` directory is in your Python path

### Performance Tips

- Use CUDA for faster inference if available
- The model works best with 1024x1024 input images
- For batch processing, you can modify the script to handle multiple images

## Model Weights

The model expects weights from the file:
```
weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth
```

Make sure this file exists before running inference.

## Example Output

The inference script will:
1. Load and preprocess the input image
2. Run the model to predict flow maps
3. Apply the flow maps to generate the retouched image
4. Save the result and optionally the flow visualization

The output will show:
- Device being used
- Image loading confirmation
- Model loading confirmation
- Save paths for results 