<div align="center">
  <h1>🌐 ChebDepth: Graph-Based Depth Estimation</h1>
  <p><i>A PyTorch pipeline for Monocular Depth Estimation using Chebyshev Graph Convolutions (ChebNet).</i></p>
</div>

## 📖 Overview
ChebDepth leverages **Superpixel Segmentation** and **Chebyshev Graph Convolutions** to estimate dense depth maps from monocular images. By framing images as region-adjacency graphs (RAGs), it learns complex, long-range dependencies across superpixels, making it highly effective at preserving structural object boundaries while remaining computationally efficient.

## ✨ Key Features
* **Graph-Based Processing:** Converts RGB images into graphs using SLIC superpixels and processes them via PyTorch Geometric's `ChebConv`.
* **Barycentric Projection:** Smoothly projects localized graph-level features back into a dense 2D image space using Delaunay triangulation.
* **Advanced Multi-Term Loss:**
  * **MSE Loss:** Core metric for depth scale.
  * **Gradient Matching Loss:** Preserves structural similarity of depth edges.
  * **Edge-Aware Smoothness:** Penalizes depth artifacts while allowing sharp transitions where real image edges exist.
  * **Graph Regularization (Dirichlet Energy):** Ensures consistency across connected superpixel representation features.
* **CLI Ready:** Configurable training pipeline straight from the terminal.

## 🚀 Getting Started

### Prerequisites
1. Python 3.8+
2. Install the necessary dependencies (we recommend a virtual environment):
```bash
pip install -r requirements.txt
```

### Dataset Structure
Your dataset directory should contain image and depth pairs:
```
data/
├── image1.jpg
├── image1_depth.png
├── image2.png
├── image2_depth.npy
└── ...
```

### Training
Train the complete pipeline with advanced losses out-of-the-box. Run the following command from your terminal:
```bash
python chebfeatdepth.py --data_dir path/to/dataset --batch_size 4 --epochs 50 --image_size 256
```

### Testing & Visualization
The pipeline automatically tests the best model at the end of training. It saves side-by-side visualizations (Input | Ground Truth | Prediction) inside an output `test_results/` folder.


## Architecture Overview

1. **Image to Graph**  
   - Input image is segmented into superpixels (SLIC) or a regular grid.  
   - Nodes: superpixel centers (or grid points).  
   - Node features: RGB values at the node location + normalized (x,y) coordinates.  
   - Edges: k‑nearest neighbors (k=8).

2. **ChebNet Feature Extractor**  
   - Multiple Chebyshev graph convolution layers (K=3).  
   - Global attention pooling to aggregate node features.  
   - Output: per‑node features (and pooled global features).

3. **Graph to Image Projection**  
   - Barycentric interpolation via Delaunay triangulation to map node features back to every pixel.

4. **Depth Head**  
   - A lightweight CNN decoder (convolutional layers with upsampling) that takes the projected feature map and predicts a depth map (sigmoid output, depth normalized to [0,1]).

5. **Training Losses**  
   - MSE + gradient matching + edge‑aware smoothness + graph regularization (Dirichlet energy).  
   - Combined to encourage accurate and smooth depth maps.

## 🏗️ Pipeline Architecture
1. `ImageToGraphConverter`: Converts RGB -> Lab Space -> SLIC Superpixels -> k-NN Graph.
2. `ChebNetFeatureExtractor`: Multi-layer Chebyshev Graph Convolutions with global attention pooling.
3. `GraphToImageProjector`: Barycentric coordinate mapping from graph nodes back to a dense grid.
4. `DepthEstimationHead`: Upsampling CNN layers that produce a normalized `[0,1]` depth map.


--------------------------------------------------------
Ishan Narayan
ishannarayan.in@gmail.com, ishan.csio21a@acsir.res.in
