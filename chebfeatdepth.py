import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, Batch
import cv2
import os
from pathlib import Path
from scipy.spatial import Delaunay
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DepthEstimationDataset(Dataset):
    """Dataset for loading images and depth maps"""
    
    def __init__(self, data_dir, image_size=(256, 256), normalize=True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalize = normalize
        
        # Find all image files (assuming common image extensions)
        self.image_files = []
        self.depth_files = []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        depth_extensions = {'.png', '.npy', '.tiff'}
        
        # Look for images and corresponding depth maps
        for img_file in self.data_dir.glob('*'):
            if img_file.suffix.lower() in image_extensions:
                # Look for corresponding depth file
                depth_candidates = [
                    self.data_dir / f"{img_file.stem}_depth{img_file.suffix}",
                    self.data_dir / f"{img_file.stem}_depth.png",
                    #self.data_dir / f"{img_file.stem}_depth.npy",
                    self.data_dir / f"{img_file.stem}{img_file.suffix}",  # same name but in different folder
                ]
                
                for depth_candidate in depth_candidates:
                    if depth_candidate.exists():
                        self.image_files.append(img_file)
                        self.depth_files.append(depth_candidate)
                        break
        
        print(f"Found {len(self.image_files)} image-depth pairs")
        
    def __len__(self):
        return len(self.image_files)
    
    def load_depth_map(self, depth_path):
        """Load depth map from file"""
        suffix = depth_path.suffix.lower()
        
        if suffix == '.npy':
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if depth is None:
                raise ValueError(f"Could not load depth map from {depth_path}")
            
            # If depth map has multiple channels, take first channel
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
        
        return depth
    
    def preprocess_depth(self, depth, target_size):
        """Preprocess depth map"""
        # Resize depth map
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize depth to [0, 1]
        if self.normalize:
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth
    
    def __getitem__(self, idx):
        try:
            # Load image
            image_path = self.image_files[idx]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1).float()
            
            # Load depth map
            depth_path = self.depth_files[idx]
            depth = self.load_depth_map(depth_path)
            depth = self.preprocess_depth(depth, self.image_size)
            depth = torch.tensor(depth).float().unsqueeze(0)  # Add channel dimension
            
            return image, depth
            
        except Exception as e:
            print(f"Error loading {self.image_files[idx]}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, *self.image_size)
            dummy_depth = torch.zeros(1, *self.image_size)
            return dummy_image, dummy_depth

class ImageToGraphConverter:
    """Convert images to graph representation for ChebNet processing"""
    
    def __init__(self, node_strategy='superpixel', num_nodes=1024, k_neighbors=8):
        self.node_strategy = node_strategy
        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors
        
    def create_grid_nodes(self, height, width):
        """Create regular grid of nodes"""
        x = np.linspace(0, 1, int(np.sqrt(self.num_nodes)))
        y = np.linspace(0, 1, int(np.sqrt(self.num_nodes)))
        xx, yy = np.meshgrid(x, y)
        nodes = np.stack([xx.flatten(), yy.flatten()], axis=1)
        return nodes
    
    def create_superpixel_nodes(self, image):
        """Create nodes using superpixel centers"""
        # Convert to LAB color space for better superpixel segmentation
        if len(image.shape) == 3 and image.shape[0] == 3:
            image_np = image.permute(1, 2, 0).cpu().numpy() if torch.is_tensor(image) else image
        else:
            image_np = image.cpu().numpy() if torch.is_tensor(image) else image
            
        lab = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # SLIC superpixels
        slic = cv2.ximgproc.createSuperpixelSLIC(lab, algorithm=cv2.ximgproc.SLICO, 
                                               region_size=20, ruler=10.0)
        slic.iterate(10)
        
        # Get superpixel centers
        labels = slic.getLabels()
        num_sp = slic.getNumberOfSuperpixels()
        
        centers = []
        for i in range(num_sp):
            mask = labels == i
            if np.sum(mask) > 0:
                y, x = np.where(mask)
                center_y, center_x = np.mean(y) / image_np.shape[0], np.mean(x) / image_np.shape[1]
                centers.append([center_x, center_y])
        
        return np.array(centers)
    
    def create_edge_index(self, nodes, k=8):
        """Create k-NN graph edges"""
        from sklearn.neighbors import kneighbors_graph
        
        # Build k-NN graph
        adj = kneighbors_graph(nodes, k, mode='connectivity', include_self=False)
        adj = adj.tocoo()
        
        # Convert to PyTorch Geometric edge index format
        edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
        
        return edge_index
    
    def extract_node_features(self, image, nodes):
        """Extract RGB and positional features for each node"""
        if len(image.shape) == 3 and image.shape[0] == 3:
            h, w = image.shape[1], image.shape[2]
            image_np = image.permute(1, 2, 0).cpu().numpy() if torch.is_tensor(image) else image
        else:
            h, w = image.shape
            image_np = image.cpu().numpy() if torch.is_tensor(image) else image
            
        features = []
        
        for node in nodes:
            x, y = node
            # Convert normalized coordinates to pixel coordinates
            px = int(x * (w - 1))
            py = int(y * (h - 1))
            
            # RGB features
            rgb = image_np[py, px] if len(image_np.shape) == 3 else [image_np[py, px]]
            
            # Positional features (normalized)
            pos_features = [x, y]
            
            # Combine features
            node_features = np.concatenate([rgb, pos_features])
            features.append(node_features)
        
        return np.array(features)
    
    def __call__(self, image):
        """Convert image to graph data"""
        h, w = image.shape[1], image.shape[2]
        
        # Create nodes based on strategy
        if self.node_strategy == 'grid':
            nodes = self.create_grid_nodes(h, w)
        elif self.node_strategy == 'superpixel':
            nodes = self.create_superpixel_nodes(image)
        else:
            raise ValueError("Node strategy must be 'grid' or 'superpixel'")
        
        # Create edges
        edge_index = self.create_edge_index(nodes, self.k_neighbors)
        
        # Extract node features
        node_features = self.extract_node_features(image, nodes)
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        pos = torch.tensor(nodes, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, pos=pos, image_size=(h, w))

class ChebNetFeatureExtractor(nn.Module):
    """ChebNet-based feature extractor for images"""
    
    def __init__(self, in_channels=5, hidden_channels=[64, 128, 256, 512], 
                 k_chebyshev=3, dropout=0.1):
        super(ChebNetFeatureExtractor, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.convs.append(ChebConv(in_channels, hidden_channels[0], K=k_chebyshev))
        self.bns.append(nn.BatchNorm1d(hidden_channels[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(ChebConv(hidden_channels[i-1], hidden_channels[i], K=k_chebyshev))
            self.bns.append(nn.BatchNorm1d(hidden_channels[i]))
        
        # Global attention pooling
        self.attention_weights = nn.Linear(hidden_channels[-1], 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply Chebyshev graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global attention pooling
        attention_scores = torch.tanh(self.attention_weights(x))
        attention_weights = F.softmax(attention_scores, dim=0)
        global_features = torch.sum(x * attention_weights, dim=0, keepdim=True)
        
        return x, global_features, attention_weights

class GraphToImageProjector:
    """Project graph features back to image space for depth estimation"""
    
    def __init__(self, method='barycentric'):
        self.method = method
    
    def barycentric_interpolation(self, graph_data, node_features, target_size):
        """Interpolate using barycentric coordinates from Delaunay triangulation"""
        h, w = target_size
        
        # Create target grid
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid_points = np.stack([x_coords.flatten() / (w-1), 
                              y_coords.flatten() / (h-1)], axis=1)
        
        # Delaunay triangulation of graph nodes
        nodes = graph_data.pos.cpu().numpy() if torch.is_tensor(graph_data.pos) else graph_data.pos
        node_features_np = node_features.detach().cpu().numpy() if torch.is_tensor(node_features) else node_features
        tri = Delaunay(nodes)
        
        # Find which triangle contains each grid point
        simplex_indices = tri.find_simplex(grid_points)
        
        # Interpolate features using barycentric coordinates
        interpolated_features = np.zeros((grid_points.shape[0], node_features_np.shape[1]))
        
        for i, simplex in enumerate(simplex_indices):
            if simplex != -1:  # Point is within triangulation
                # Get triangle vertices
                vertices = tri.simplices[simplex]
                # Get barycentric coordinates
                bcoords = tri.transform[simplex, :2].dot(
                    grid_points[i] - tri.transform[simplex, 2])
                bcoords = np.append(bcoords, 1 - bcoords.sum())
                
                # Interpolate features
                interpolated_features[i] = np.sum(
                    node_features_np[vertices] * bcoords[:, np.newaxis], axis=0)
            else:
                # Use nearest node for points outside triangulation
                distances = np.linalg.norm(nodes - grid_points[i], axis=1)
                nearest_node = np.argmin(distances)
                interpolated_features[i] = node_features_np[nearest_node]
        
        # Reshape to image format
        feature_image = interpolated_features.reshape(h, w, -1)
        return torch.tensor(feature_image.transpose(2, 0, 1), dtype=torch.float32)
    
    def __call__(self, graph_data, node_features, target_size):
        if self.method == 'barycentric':
            return self.barycentric_interpolation(graph_data, node_features, target_size)
        else:
            raise ValueError("Interpolation method not supported")

class DepthEstimationHead(nn.Module):
    """Depth estimation head that uses ChebNet features"""
    
    def __init__(self, in_channels, hidden_channels=[256, 128, 64, 32], output_channels=1):
        super(DepthEstimationHead, self).__init__()
        
        layers = []
        channels = [in_channels] + hidden_channels
        
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ])
            
            # Add upsampling in later layers
            if i >= len(channels) - 3:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # Final depth prediction layer
        layers.append(nn.Conv2d(channels[-1], output_channels, kernel_size=1))
        # Use Sigmoid since we normalized depth to [0,1]
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.decoder(x))

class ChebNetDepthPipeline(nn.Module):
    """Complete pipeline for depth estimation using ChebNet features"""
    
    def __init__(self, config=None):
        super(ChebNetDepthPipeline, self).__init__()
        
        if config is None:
            config = {
                'graph_converter': {
                    'node_strategy': 'superpixel',
                    'num_nodes': 1024,
                    'k_neighbors': 8
                },
                'feature_extractor': {
                    'in_channels': 5,  # RGB + x,y coordinates
                    'hidden_channels': [64, 128, 256, 512],
                    'k_chebyshev': 3,
                    'dropout': 0.1
                },
                'depth_head': {
                    'in_channels': 512,
                    'hidden_channels': [256, 128, 64, 32],
                    'output_channels': 1
                }
            }
        
        self.graph_converter = ImageToGraphConverter(**config['graph_converter'])
        self.feature_extractor = ChebNetFeatureExtractor(**config['feature_extractor'])
        self.projector = GraphToImageProjector(method='barycentric')
        self.depth_head = DepthEstimationHead(**config['depth_head'])
        
    def forward(self, image):
        batch_size = image.shape[0]
        
        # Process each image in batch separately (graph conversion doesn't batch easily)
        batch_depth_preds = []
        batch_features = []
        
        for i in range(batch_size):
            single_image = image[i]
            
            # Convert image to graph
            graph_data = self.graph_converter(single_image)
            
            # Move graph data to same device as image
            device = single_image.device
            graph_data = graph_data.to(device)
            
            # Extract features using ChebNet
            node_features, global_features, attention_weights = self.feature_extractor(graph_data)
            
            # Project graph features back to image space
            h, w = single_image.shape[1], single_image.shape[2]
            image_features = self.projector(graph_data, node_features, (h, w))
            
            # Move projected features back to device and estimate depth
            image_features = image_features.to(device)
            depth_pred = self.depth_head(image_features.unsqueeze(0))
            batch_depth_preds.append(depth_pred)
            
            batch_features.append({
                'node_features': node_features,
                'global_features': global_features,
                'attention_weights': attention_weights,
                'edge_index': graph_data.edge_index.to(device)
            })
        
        # Stack batch predictions
        depth_preds = torch.cat(batch_depth_preds, dim=0)
        
        return depth_preds, batch_features

class DepthEstimationTrainer:
    """Trainer class for depth estimation model"""
    
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Loss function combining RMSE and gradient loss for better depth estimation
        self.depth_criterion = nn.MSELoss()
        
    def gradient_loss(self, pred, target):
        """Gradient matching loss (structural similarity of depth edges)"""
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        loss_dx = torch.abs(pred_dx - target_dx).mean()
        loss_dy = torch.abs(pred_dy - target_dy).mean()
        
        return loss_dx + loss_dy

    def edge_aware_smoothness_loss(self, pred, image):
        """Edge-aware smoothness loss (depth gradient weighted by inverse image gradient)"""
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        # Image gradients (mean across color channels)
        image_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
        image_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)
        
        # Exponential weight to allow depth jumps where image has edges
        weights_x = torch.exp(-image_dx)
        weights_y = torch.exp(-image_dy)
        
        smoothness_x = pred_dx * weights_x
        smoothness_y = pred_dy * weights_y
        
        return smoothness_x.mean() + smoothness_y.mean()

    def cheb_graph_loss(self, batch_features):
        """Graph Regularization / Dirichlet Energy for ChebNet features"""
        graph_loss = 0.0
        for b in batch_features:
            node_features = b['node_features']  # [N, C]
            edge_index = b['edge_index']        # [2, E]
            
            src, dst = edge_index[0], edge_index[1]
            diff = node_features[src] - node_features[dst]
            
            # Smoothness constraint: connected nodes should have similar features
            loss = torch.norm(diff, p=2, dim=1).mean()
            graph_loss += loss
            
        return graph_loss / max(len(batch_features), 1)
    
    def compute_loss(self, pred, target, image, batch_features):
        """Compute advanced combined depth estimation loss"""
        mse_loss = self.depth_criterion(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        smooth_loss = self.edge_aware_smoothness_loss(pred, image)
        cheb_loss = self.cheb_graph_loss(batch_features)
        
        # Weighted combination
        total_loss = mse_loss + 0.5 * grad_loss + 0.1 * smooth_loss + 0.05 * cheb_loss
        return total_loss, mse_loss, grad_loss, smooth_loss, cheb_loss
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, depths) in enumerate(progress_bar):
            images, depths = images.to(self.device), depths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            depth_preds, batch_features = self.model(images)
            
            # Compute loss
            loss, mse_loss, grad_loss, smooth_loss, cheb_loss = self.compute_loss(
                depth_preds, depths, images, batch_features
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'L': f'{loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Grad': f'{grad_loss.item():.4f}',
                'Sm': f'{smooth_loss.item():.4f}',
                'Ch': f'{cheb_loss.item():.4f}'
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, depths in tqdm(self.val_loader, desc="Validation"):
                images, depths = images.to(self.device), depths.to(self.device)
                
                depth_preds, batch_features = self.model(images)
                loss, _, _, _, _ = self.compute_loss(depth_preds, depths, images, batch_features)
                val_loss += loss.item()
                
                all_preds.append(depth_preds.cpu())
                all_targets.append(depths.cpu())
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))
        mae = F.l1_loss(all_preds, all_targets)
        
        return val_loss / len(self.val_loader), rmse.item(), mae.item()
    
    def train(self, num_epochs, save_path="chebnet_depth_model.pth"):
        """Full training loop"""
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss, rmse, mae = self.validate()
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, save_path)
                print(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Plot progress
            self.plot_training_progress(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def plot_training_progress(self, train_losses, val_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.close()

def test_model(model, test_loader, device, save_dir="test_results"):
    """Test the trained model and visualize results"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, depths) in enumerate(tqdm(test_loader, desc="Testing")):
            images, depths = images.to(device), depths.to(device)
            
            depth_preds, features = model(images)
            
            all_preds.append(depth_preds.cpu())
            all_targets.append(depths.cpu())
            
            # Visualize first few samples
            if i < 5:  # Save first 5 test samples
                for j in range(min(images.shape[0], 3)):  # First 3 in batch
                    visualize_sample(
                        images[j].cpu(), 
                        depths[j].cpu(), 
                        depth_preds[j].cpu(),
                        save_path=os.path.join(save_dir, f'sample_{i}_{j}.png')
                    )
    
    # Compute final metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))
    mae = F.l1_loss(all_preds, all_targets)
    
    print(f"\nFinal Test Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return rmse.item(), mae.item()

def visualize_sample(image, target_depth, pred_depth, save_path):
    """Visualize a single test sample"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image.shape[0] == 3:
        axes[0].imshow(image.permute(1, 2, 0))
    else:
        axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth depth
    axes[1].imshow(target_depth.squeeze(), cmap='plasma')
    axes[1].set_title('Ground Truth Depth')
    axes[1].axis('off')
    
    # Predicted depth
    axes[2].imshow(pred_depth.squeeze(), cmap='plasma')
    axes[2].set_title('Predicted Depth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

import argparse

def main():
    """Main function to run training and testing"""
    parser = argparse.ArgumentParser(description="Train and Test ChebNet Depth Estimation Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--image_size', type=int, default=256, help="Image resolution (size x size)")
    parser.add_argument('--workers', type=int, default=2, help="Number of data loading workers")
    parser.add_argument('--save_path', type=str, default="best_chebnet_depth_model.pth", help="Path to save the best model")
    
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    IMAGE_SIZE = (args.image_size, args.image_size)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Configuration: Batch={BATCH_SIZE}, Epochs={NUM_EPOCHS}, ImgSize={IMAGE_SIZE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' does not exist.")
        return
        
    # Load dataset
    print("Loading dataset...")
    dataset = DepthEstimationDataset(DATA_DIR, image_size=IMAGE_SIZE)
    
    if len(dataset) == 0:
        print("Error: No image-depth pairs found in the given directory.")
        return
        
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Use fixed generator seed for deterministic split across tests
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = ChebNetDepthPipeline()
    
    # Initialize trainer
    trainer = DepthEstimationTrainer(model, train_loader, val_loader, DEVICE)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = trainer.train(NUM_EPOCHS, args.save_path)
    
    # Load best model for testing
    checkpoint = torch.load(args.save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    print("\nTesting model...")
    rmse, mae = test_model(model, test_loader, DEVICE, save_dir="test_results")
    
    print(f"\n=== Final Results ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

if __name__ == "__main__":
    main()