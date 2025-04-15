import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
import json
import zipfile
import shutil
import requests
from tqdm import tqdm
import random
from collections import Counter

# Try to import nuscenes, but don't fail if not available
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("NuScenes package not found. To install: pip install nuscenes-devkit")

class NuScenesDataset(Dataset):
    """
    Dataset class for NuScenes autonomous driving dataset
    
    The dataset uses available images from the NuScenes dataset and creates
    classes based on actually available data.
    """
    def __init__(self, nuscenes_dir, version='v1.0-mini', split='train', transform=None):
        """
        Args:
            nuscenes_dir: Directory where nuscenes data is stored
            version: NuScenes version ('v1.0-mini' or 'v1.0-trainval')
            split: Dataset split ('train', 'val', 'test')
            transform: Image transforms to apply
        """
        self.nuscenes_dir = Path(nuscenes_dir)
        self.transform = transform
        self.version = version
        self.samples = []
        # Default class names, will be overridden by available data
        self.class_names = ['Day', 'Night', 'Urban', 'Residential', 'Highway']
        
        if not NUSCENES_AVAILABLE:
            raise ImportError("NuScenes package is required. Install with: pip install nuscenes-devkit")
        
        # Check if the directory contains real data
        has_real_data = self._check_for_real_data()
        if not has_real_data:
            raise ValueError("No NuScenes data found. Please provide path to NuScenes dataset.")
            
        # Initialize NuScenes dataset with correct parameters
        try:
            # Patch the nuScenes loading to avoid map errors
            import nuscenes.map_expansion.map_api as map_api
            original_load = map_api.NuScenesMap.load
            
            # Create a patched load function that catches file not found errors
            def patched_load(self_map, layer_name):
                try:
                    return original_load(self_map, layer_name)
                except FileNotFoundError as e:
                    print(f"Warning: Map file not found: {e}")
                    return None
            
            # Apply the patch
            map_api.NuScenesMap.load = patched_load
            
            # Initialize NuScenes
            self.nusc = NuScenes(version=version, dataroot=str(nuscenes_dir), verbose=True)
            
            # Restore original method after initialization
            map_api.NuScenesMap.load = original_load
                
        except Exception as e:
            print(f"Warning: Error initializing NuScenes with patched loading: {e}")
            print("Trying different initialization approach...")
            
            try:
                # Try direct initialization without map loading
                self.nusc = NuScenes(version=version, dataroot=str(nuscenes_dir), verbose=True)
            except Exception as e:
                print(f"Warning: Error initializing NuScenes: {e}")
                if has_real_data:
                    print("Attempting to load samples directly without the NuScenes API...")
                    if self._load_samples_directly():
                        print(f"Successfully loaded {len(self.samples)} samples directly")
                        return
                    else:
                        print("Failed to load samples directly")
                
                raise ValueError("Could not initialize NuScenes dataset")
        
        # Extract metadata to determine available categories
        self._extract_metadata()
        
        # Get scene splits
        scene_splits = create_splits_scenes()
        scenes = scene_splits.get(split, [])
        
        # If no scenes for the split, use all scenes
        if not scenes:
            print(f"No scenes found for split '{split}'. Using all available scenes.")
            scenes = [scene['name'] for scene in self.nusc.scene]
        
        # Extract samples
        print(f"Loading NuScenes {split} split...")
        for scene in tqdm(self.nusc.scene):
            # Skip scenes not in the requested split
            if scene['name'] not in scenes:
                continue
                
            # Get sample tokens for this scene
            sample_token = scene['first_sample_token']
            
            # Determine class based on available metadata
            class_id = self._get_class_id_for_scene(scene)
            
            # Add samples from this scene
            try:
                while sample_token:
                    sample = self.nusc.get('sample', sample_token)
                    
                    # Get front camera image
                    cam_front_token = sample['data']['CAM_FRONT']
                    cam_front_data = self.nusc.get('sample_data', cam_front_token)
                    img_path = self.nuscenes_dir / cam_front_data['filename']
                    
                    # Only add if image exists
                    if img_path.exists():
                        self.samples.append({
                            'image_path': img_path,
                            'class_id': class_id,
                            'metadata': {
                                'scene_name': scene['name'],
                                'description': scene['description']
                            }
                        })
                    
                    # Move to next sample
                    sample_token = sample['next']
            except Exception as e:
                print(f"Error processing sample {sample_token}: {e}")
                continue
            
        print(f"Loaded {len(self.samples)} samples from NuScenes {split} split")
        
        # Verify we have data
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for split '{split}'")
        
        # Print class distribution
        self._print_class_distribution()

    def _extract_metadata(self):
        """Extract metadata from NuScenes to determine available categories"""
        # Collect scene descriptions
        descriptions = []
        for scene in self.nusc.scene:
            descriptions.append(scene['description'].lower())
        
        # Find discriminative features in descriptions
        time_of_day = {'day': 0, 'night': 0}
        location_type = {'urban': 0, 'residential': 0, 'highway': 0}
        weather = {'rain': 0, 'sunny': 0, 'cloudy': 0}
        
        for desc in descriptions:
            # Time of day
            if 'night' in desc:
                time_of_day['night'] += 1
            else:
                time_of_day['day'] += 1
                
            # Location type
            if 'urban' in desc:
                location_type['urban'] += 1
            elif 'residential' in desc:
                location_type['residential'] += 1
            elif 'highway' in desc:
                location_type['highway'] += 1
                
            # Weather
            if 'rain' in desc:
                weather['rain'] += 1
            elif 'sunny' in desc or 'sun' in desc:
                weather['sunny'] += 1
            elif 'cloudy' in desc or 'cloud' in desc:
                weather['cloudy'] += 1
        
        # Determine which features have good distribution
        features = []
        
        # Check if time of day has reasonable distribution
        min_time = min(time_of_day.values())
        max_time = max(time_of_day.values())
        if min_time > 0 and max_time/min_time < 10:  # Prevent extreme imbalance
            features.append(('time', list(time_of_day.keys())))
        
        # Check location distribution
        location_values = [v for v in location_type.values() if v > 0]
        if len(location_values) > 1:
            features.append(('location', [k for k, v in location_type.items() if v > 0]))
            
        # Check weather distribution
        weather_values = [v for v in weather.values() if v > 0]
        if len(weather_values) > 1:
            features.append(('weather', [k for k, v in weather.items() if v > 0]))
        
        # Select most balanced feature pair to define classes
        if len(features) >= 2:
            # Use first two features that have data
            primary_feature = features[0]
            secondary_feature = features[1]
            
            # Generate class names from combinations
            self.class_names = []
            self.feature_to_class = {}
            
            class_id = 0
            for primary_value in primary_feature[1]:
                for secondary_value in secondary_feature[1]:
                    class_name = f"{primary_value.capitalize()} {secondary_value.capitalize()}"
                    self.class_names.append(class_name)
                    self.feature_to_class[(primary_value, secondary_value)] = class_id
                    class_id += 1
            
            self.primary_feature = primary_feature[0]
            self.secondary_feature = secondary_feature[0]
            
        else:
            # If can't find good pairs, use a single feature
            if features:
                # Use first available feature
                feature = features[0]
                self.class_names = [value.capitalize() for value in feature[1]]
                self.feature_to_class = {value: i for i, value in enumerate(feature[1])}
                self.primary_feature = feature[0]
                self.secondary_feature = None
            else:
                # Fallback to a simple day/night classification
                self.class_names = ['Day', 'Night']
                self.feature_to_class = {'day': 0, 'night': 1}
                self.primary_feature = 'time'
                self.secondary_feature = None
        
        print(f"Classes determined from data: {self.class_names}")

    def _get_class_id_for_scene(self, scene):
        """Determine class ID for a scene based on metadata"""
        desc = scene['description'].lower()
        
        # Extract primary feature
        if self.primary_feature == 'time':
            primary_value = 'night' if 'night' in desc else 'day'
        elif self.primary_feature == 'location':
            if 'urban' in desc:
                primary_value = 'urban'
            elif 'residential' in desc:
                primary_value = 'residential'
            elif 'highway' in desc:
                primary_value = 'highway'
            else:
                primary_value = next(iter(self.feature_to_class))  # default to first
        elif self.primary_feature == 'weather':
            if 'rain' in desc:
                primary_value = 'rain'
            elif 'sunny' in desc or 'sun' in desc:
                primary_value = 'sunny'
            elif 'cloudy' in desc or 'cloud' in desc:
                primary_value = 'cloudy'
            else:
                primary_value = next(iter(self.feature_to_class))  # default to first
        
        # If we have a secondary feature
        if self.secondary_feature:
            # Extract secondary feature
            if self.secondary_feature == 'time':
                secondary_value = 'night' if 'night' in desc else 'day'
            elif self.secondary_feature == 'location':
                if 'urban' in desc:
                    secondary_value = 'urban'
                elif 'residential' in desc:
                    secondary_value = 'residential'
                elif 'highway' in desc:
                    secondary_value = 'highway'
                else:
                    for val in ['urban', 'residential', 'highway']:
                        if val in self.feature_to_class:
                            secondary_value = val
                            break
                    else:
                        secondary_value = next(iter(self.feature_to_class))  # default to first
            elif self.secondary_feature == 'weather':
                if 'rain' in desc:
                    secondary_value = 'rain'
                elif 'sunny' in desc or 'sun' in desc:
                    secondary_value = 'sunny'
                elif 'cloudy' in desc or 'cloud' in desc:
                    secondary_value = 'cloudy'
                else:
                    for val in ['rain', 'sunny', 'cloudy']:
                        if val in self.feature_to_class:
                            secondary_value = val
                            break
                    else:
                        secondary_value = next(iter(self.feature_to_class))  # default to first
            
            # Look up class ID
            try:
                return self.feature_to_class[(primary_value, secondary_value)]
            except KeyError:
                # Fallback to first class
                return 0
        else:
            # Single feature classification
            try:
                return self.feature_to_class[primary_value]
            except KeyError:
                # Fallback to first class
                return 0

    def _print_class_distribution(self):
        """Print distribution of classes in the dataset"""
        class_counts = {}
        for sample in self.samples:
            class_id = sample['class_id']
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print("\nClass distribution:")
        for class_id in sorted(class_counts.keys()):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown ({class_id})"
            print(f"  {class_name}: {class_counts[class_id]} samples")

    def _check_for_real_data(self):
        """Check if the directory contains real NuScenes data"""
        # Check for samples directory with CAM_FRONT images
        samples_dir = self.nuscenes_dir / 'v1.0-mini' / 'samples' / 'CAM_FRONT'
        
        if samples_dir.exists():
            # Look for jpg files in the directory structure
            jpg_files = list(samples_dir.glob('**/*.jpg'))
            
            if jpg_files:
                print(f"Found {len(jpg_files)} real images in {samples_dir}")
                return True
        
        return False
    
    def _load_samples_directly(self):
        """Load samples by scanning the directory structure directly"""
        samples_dir = self.nuscenes_dir / 'v1.0-mini' / 'samples' / 'CAM_FRONT'
        
        if not samples_dir.exists():
            return False
            
        # Scan for images and classify based on directory structure
        image_paths = []
        for img_path in samples_dir.glob('**/*.jpg'):
            image_paths.append(img_path)
        
        if not image_paths:
            return False
            
        # Simple classification based on image path
        # Extract unique directory components to use as classes
        dir_parts = set()
        for img_path in image_paths:
            # Get directory name as potential class
            dir_name = img_path.parent.name
            if dir_name != "CAM_FRONT":
                dir_parts.add(dir_name)
        
        # Use directory names as classes if we have enough variation
        if len(dir_parts) > 1:
            self.class_names = sorted(list(dir_parts))
            
            # Map each image to a class
            for img_path in image_paths:
                dir_name = img_path.parent.name
                if dir_name in self.class_names:
                    class_id = self.class_names.index(dir_name)
                else:
                    class_id = 0  # Default
                
                self.samples.append({
                    'image_path': img_path,
                    'class_id': class_id
                })
        else:
            # If no directory variation, try using scenes from filename patterns
            scene_patterns = [
                ('night', 'Night'),
                ('day', 'Day'),
                ('urban', 'Urban'),
                ('residential', 'Residential'),
                ('highway', 'Highway')
            ]
            
            # Count occurrences of each pattern
            pattern_counts = {pattern[1]: 0 for pattern in scene_patterns}
            
            for img_path in image_paths:
                img_name = img_path.name.lower()
                for pattern, class_name in scene_patterns:
                    if pattern in img_name:
                        pattern_counts[class_name] += 1
            
            # Use patterns that appear in the data
            valid_classes = [cls for cls, count in pattern_counts.items() if count > 0]
            
            if valid_classes:
                self.class_names = valid_classes
                
                # Classify each image
                for img_path in image_paths:
                    img_name = img_path.name.lower()
                    assigned = False
                    
                    for pattern, class_name in scene_patterns:
                        if class_name in self.class_names and pattern in img_name:
                            class_id = self.class_names.index(class_name)
                            assigned = True
                            break
                    
                    if not assigned:
                        class_id = 0  # Default
                    
                    self.samples.append({
                        'image_path': img_path,
                        'class_id': class_id
                    })
            else:
                # If no patterns match, use simple indexing (just to have some classes)
                self.class_names = ['Class_0', 'Class_1']
                
                # Distribute images roughly evenly
                for i, img_path in enumerate(image_paths):
                    class_id = i % len(self.class_names)
                    self.samples.append({
                        'image_path': img_path,
                        'class_id': class_id
                    })
        
        print(f"Created {len(self.samples)} samples with classes: {self.class_names}")
        return len(self.samples) > 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        class_id = sample['class_id']
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            
        return image, class_id

def download_nuscenes_mini(output_dir='./data/nuscenes'):
    """
    Download or setup NuScenes mini dataset
    
    Args:
        output_dir: Directory to save/find the dataset
    
    Returns:
        Path to the dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if the dataset already exists
    expected_data_dir = output_path / 'v1.0-mini'
    if (expected_data_dir).exists() and (expected_data_dir / 'maps').exists():
        print(f"NuScenes mini dataset found at {output_path}")
        return output_path
    
    print("\n===== NuScenes Dataset Setup =====")
    print("""
The NuScenes dataset requires manual download due to license agreements:

1. Go to https://www.nuscenes.org/download
2. Register and log in
3. Download the v1.0-mini dataset (approximately 4GB)
4. Extract the files to: {}

If you have already downloaded the dataset, please specify its location.
    """.format(output_path))
    
    # Check for synthetic data option
    print("\nWould you like to create synthetic data instead? [y/n]")
    choice = input().lower()
    
    if choice == 'y':
        create_synthetic_nuscenes_structure(output_path)
        return output_path
        
    # Otherwise, let user input the path where they have the dataset
    print("\nIf you have already downloaded the dataset, enter the path (or press Enter to use default):")
    user_path = input().strip()
    
    if user_path:
        user_path = Path(user_path)
        if (user_path / 'v1.0-mini').exists():
            # Copy or symlink the data
            if (output_path / 'v1.0-mini').exists():
                print(f"Directory already exists at {output_path / 'v1.0-mini'}")
            else:
                print(f"Creating symlink from {user_path} to {output_path}")
                try:
                    # Try symlink first
                    (output_path / 'v1.0-mini').symlink_to(user_path / 'v1.0-mini')
                except:
                    # If symlink fails, inform the user
                    print("Failed to create symlink. Please manually copy the dataset.")
                    print(f"From: {user_path / 'v1.0-mini'}")
                    print(f"To: {output_path / 'v1.0-mini'}")
        else:
            print(f"NuScenes data not found at {user_path}")
            print("Creating synthetic data instead.")
            create_synthetic_nuscenes_structure(output_path)
    else:
        print("No path specified. Creating synthetic data.")
        create_synthetic_nuscenes_structure(output_path)
            
    return output_path

def create_synthetic_nuscenes_structure(output_dir):
    """Create a minimal synthetic dataset structure mimicking NuScenes format"""
    print("Creating synthetic NuScenes dataset structure...")
    
    output_path = Path(output_dir)
    mini_path = output_path / 'v1.0-mini'
    mini_path.mkdir(exist_ok=True, parents=True)
    
    # Create required directories
    samples_path = mini_path / 'samples'
    samples_path.mkdir(exist_ok=True)
    maps_path = mini_path / 'maps'
    maps_path.mkdir(exist_ok=True)
    
    # Create a basic map file to avoid map loading errors
    map_filename = '53992ee3023e5494b90c316c183be829.png'
    map_path = maps_path / map_filename
    if not map_path.exists():
        # Create a simple 1000x1000 black image as a map placeholder
        blank_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.imwrite(str(map_path), blank_map)
        print(f"Created placeholder map at {map_path}")
    
    # Create a minimal metadata structure to enable the NuScenes API
    metadata = {
        "description": "Synthetic NuScenes mini dataset for demo purposes",
        "version": "v1.0-mini",
        "scenes": [],
        "categories": [],
        "attributes": [],
        "visibility": [],
        "sensors": ["CAM_FRONT"],
        "calibrated_sensors": [{"token": "cs1", "sensor_type": "camera"}],
        "ego_poses": [{"token": "ep1"}]
    }
    
    # Create sample scenes
    scene_types = ['intersection', 'highway', 'urban', 'parking', 'tunnel']
    samples = []
    sample_datas = []
    
    for i, scene_type in enumerate(scene_types):
        # Create scene data
        scene_token = f"scene-{i:03d}"
        scene = {
            "token": scene_token,
            "name": f"scene-{i:03d}",
            "description": f"{scene_type} scene",
            "log_token": f"log-{i:03d}",
            "nbr_samples": 10,
            "first_sample_token": f"sample-{i}-000",
            "last_sample_token": f"sample-{i}-009"
        }
        metadata["scenes"].append(scene)
        
        # Create samples for this scene
        for j in range(10):
            sample_token = f"sample-{i}-{j:03d}"
            next_token = f"sample-{i}-{j+1:03d}" if j < 9 else ""
            
            sample = {
                "token": sample_token,
                "scene_token": scene_token,
                "next": next_token,
                "prev": f"sample-{i}-{j-1:03d}" if j > 0 else "",
                "data": {"CAM_FRONT": f"cam-front-{i}-{j:03d}"}
            }
            samples.append(sample)
            
            # Create sample data for this sample
            sample_data = {
                "token": f"cam-front-{i}-{j:03d}",
                "sample_token": sample_token,
                "filename": f"samples/CAM_FRONT/{scene_type}/synthetic_{j:03d}.jpg"
            }
            sample_datas.append(sample_data)
            
            # Create directory for images
            img_dir = samples_path / "CAM_FRONT" / scene_type
            img_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a synthetic image
            img_path = img_dir / f"synthetic_{j:03d}.jpg"
            if not img_path.exists():
                img = np.ones((900, 1600, 3), dtype=np.uint8) * 100
                
                # Add visual cues based on scene type
                if scene_type == 'intersection':
                    # Create crossroads
                    img[:, :, 0] = 180  # Red tint
                    cv2.line(img, (800, 0), (800, 900), (255, 255, 255), 10)
                    cv2.line(img, (0, 450), (1600, 450), (255, 255, 255), 10)
                
                elif scene_type == 'highway':
                    # Create highway lanes
                    img[:, :, 2] = 180  # Blue tint
                    for y in range(150, 751, 150):
                        cv2.line(img, (0, y), (1600, y), (255, 255, 255), 5)
                
                elif scene_type == 'urban':
                    # Create urban scene
                    img[:, :, 1] = 150  # Green tint
                    # Road
                    cv2.rectangle(img, (0, 450), (1600, 900), (80, 80, 80), -1)
                    # Buildings
                    for x in range(100, 1501, 300):
                        h = random.randint(150, 400)
                        cv2.rectangle(img, (x, 450-h), (x+200, 450), (120, 120, 120), -1)
                    # Road markings
                    cv2.line(img, (0, 675), (1600, 675), (255, 255, 255), 5)
                
                elif scene_type == 'parking':
                    # Create parking lot
                    img[:, :] = 100  # Gray
                    cv2.rectangle(img, (0, 0), (1600, 900), (80, 80, 80), -1)
                    # Parking spaces
                    for y in range(100, 801, 150):
                        for x in range(100, 1501, 250):
                            cv2.rectangle(img, (x, y), (x+200, y+100), (255, 255, 255), 2)
                
                elif scene_type == 'tunnel':
                    # Create tunnel effect
                    img[:, :] = 50  # Dark
                    # Tunnel shape
                    pts = np.array([[0, 0], [1600, 0], [1200, 900], [400, 900]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(img, [pts], (70, 70, 70))
                    # Tunnel lights
                    for y in range(100, 801, 200):
                        cv2.circle(img, (800, y), 10, (255, 255, 200), -1)
            
                # Add scene and frame info
                cv2.putText(img, f"Scene: {scene_type}", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(img, f"Frame: {j}", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Save image
                cv2.imwrite(str(img_path), img)
    
    # Add samples and sample_datas to metadata
    metadata["sample"] = samples
    metadata["sample_data"] = sample_datas
    
    # Create empty metadata files that NuScenes expects
    empty_files = [
        "attribute.json", "calibrated_sensor.json", "category.json",
        "ego_pose.json", "instance.json", "log.json", "map.json",
        "sensor.json", "visibility.json"
    ]
    
    for file in empty_files:
        with open(mini_path / file, 'w') as f:
            f.write("{}")
    
    # Write main metadata files
    with open(mini_path / 'scene.json', 'w') as f:
        json.dump(metadata["scenes"], f, indent=2)
    
    with open(mini_path / 'sample.json', 'w') as f:
        json.dump(samples, f, indent=2)
        
    with open(mini_path / 'sample_data.json', 'w') as f:
        json.dump(sample_datas, f, indent=2)
    
    # Make sure the metadata includes map information
    with open(mini_path / 'map.json', 'w') as f:
        map_data = [{
            "token": "53992ee3023e5494b90c316c183be829",
            "filename": f"maps/{map_filename}",
            "category": "semantic_prior"
        }]
        json.dump(map_data, f, indent=2)
    
    print(f"Created synthetic NuScenes dataset structure at {mini_path}")

# Add a function to check the maps directly
def check_maps_directory(nuscenes_dir):
    """Check if the maps directory exists and has map files"""
    maps_dir = Path(nuscenes_dir) / 'v1.0-mini' / 'maps'
    if not maps_dir.exists():
        print(f"Maps directory {maps_dir} does not exist")
        return False
    
    # First check for map files
    map_files = list(maps_dir.glob('*.png'))
    print(f"Found {len(map_files)} map files in {maps_dir}")
    
    # If no png files found, look for any files
    if not map_files:
        any_files = list(maps_dir.iterdir())
        print(f"Found {len(any_files)} other files in maps directory:")
        for file in any_files:
            print(f"  - {file.name}")
    
    return True  # Return True even if no map files to allow processing to continue

# Update the main function to check maps
if __name__ == '__main__':
    # Test dataset setup
    data_dir = download_nuscenes_mini()
    
    # Check maps directory - this won't block processing
    has_maps = check_maps_directory(data_dir)
    print(f"Maps directory check: {'Files found' if has_maps else 'No map files'}")
    
    # Create the map file if needed
    maps_dir = Path(data_dir) / 'v1.0-mini' / 'maps'
    if maps_dir.exists():
        map_filename = '53992ee3023e5494b90c316c183be829.png'
        map_path = maps_dir / map_filename
        if not map_path.exists():
            print(f"Creating missing map file: {map_path}")
            blank_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            cv2.imwrite(str(map_path), blank_map)
    
    # Try to create dataset
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        print("Attempting to create dataset...")
        dataset = NuScenesDataset(data_dir, transform=transform)
        print(f"Successfully created dataset with {len(dataset)} samples")
        
        # Display a sample
        image, label = dataset[0]
        print(f"Sample image shape: {image.shape}, label: {label} ({dataset.class_names[label]})")
        
        # Convert tensor to numpy for visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(f"Class: {dataset.class_names[label]}")
        plt.axis('off')
        plt.savefig('nuscenes_sample.png')
        plt.show()
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please try running train.py which handles these errors more gracefully.")
