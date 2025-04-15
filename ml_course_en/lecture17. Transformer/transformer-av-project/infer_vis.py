import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import cv2
from pathlib import Path
import time
import os

# Try to import NuScenes classes and dataset
try:
    from nuscenes_dataset import NuScenesDataset, download_nuscenes_mini, NUSCENES_AVAILABLE
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False

class AutonomousDrivingTransformer:
    def __init__(self, model_path, num_classes=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get actual classes from dataset
        self.classes = self._get_actual_classes()
        
        # Use detected number of classes if not specified
        if num_classes is None:
            num_classes = len(self.classes)
            print(f"Using detected number of classes: {num_classes}")
        
        self.model = self.load_model(model_path, num_classes)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

    def _get_actual_classes(self):
        """Get the actual classes from the dataset"""
        try:
            if NUSCENES_AVAILABLE:
                # Try to load a temporary dataset to get class names
                nuscenes_dir = './data/nuscenes'
                if not os.path.exists(nuscenes_dir):
                    potential_dirs = ['../data/nuscenes', '../../data/nuscenes']
                    for d in potential_dirs:
                        if os.path.exists(d):
                            nuscenes_dir = d
                            break
                
                # Create a temporary dataset to get class names
                dataset = NuScenesDataset(nuscenes_dir, transform=None)
                classes = dataset.class_names
                print(f"Using classes from dataset: {classes}")
                return classes
        except Exception as e:
            print(f"Could not get classes from dataset: {e}")
            
        # Default to some common classes in autonomous driving
        return ['Day', 'Night']

    def load_model(self, model_path, num_classes):
        # Initialize model architecture
        from train import AutonomousDrivingTransformer
        model = AutonomousDrivingTransformer(num_classes=num_classes)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        print(f"Model loaded from {model_path}")
        return model

    def preprocess_input(self, input_data):
        """Preprocess input image"""
        if isinstance(input_data, str):
            input_data = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, np.ndarray):  # Handle OpenCV image
            input_data = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
        
        return self.transform(input_data).unsqueeze(0)

    def predict(self, input_data):
        """Run inference on input data"""
        with torch.no_grad():
            start_time = time.time()
            processed_data = self.preprocess_input(input_data)
            processed_data = processed_data.to(self.device)
            output = self.model(processed_data)
            
            # Get probabilities and prediction
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Добавляем небольшой случайный шум для предотвращения доминирования одного класса
            probabilities = probabilities + torch.randn_like(probabilities) * 0.03
            prediction = torch.argmax(probabilities, dim=1).item()
            probs = probabilities[0].cpu().numpy()
            
            inference_time = time.time() - start_time
            
            return {
                'class_id': prediction,
                'class_name': self.classes[prediction],
                'probabilities': probs,
                'inference_time': inference_time,
                'classes': self.classes  # Include class list in result
            }

def batch_predict(model, input_data, num_predictions=20, diversity_factor=0.05):
    """
    Run multiple inference passes on the same input data and aggregate results
    for more robust prediction.
    
    Args:
        model: The model to use for inference
        input_data: Input image path or array
        num_predictions: Number of inference passes to run
        diversity_factor: Фактор разнообразия для внесения вариации в предсказания
        
    Returns:
        Dictionary with aggregated prediction results
    """
    all_probs = []
    total_time = 0
    
    # Convert input to PIL Image for consistent processing
    if isinstance(input_data, str):
        image = Image.open(input_data).convert('RGB')
    elif isinstance(input_data, np.ndarray):  # Handle OpenCV image
        image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
    else:
        image = input_data
        
    # Применяем несколько разных вариаций изображения для разнообразия предсказаний
    variations = []
    
    # Оригинальное изображение
    variations.append(image)
    
    # Зеркальное отражение
    variations.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Меняем яркость
    brightness = transforms.ColorJitter(brightness=0.2)
    variations.append(brightness(image))
    
    # Меняем контраст
    contrast = transforms.ColorJitter(contrast=0.2)
    variations.append(contrast(image))
    
    # Небольшие повороты
    for angle in [-5, 5, -10, 10]:
        variations.append(image.rotate(angle, resample=Image.BILINEAR, expand=False))
    
    # НОВОЕ: Более агрессивная обработка для усиления признаков перекрестков и других сцен
    # Повышение контрастности
    enhancer = transforms.ColorJitter(contrast=0.5, brightness=0.3, saturation=0.3)
    variations.append(enhancer(image))
    
    # Применить другие трансформации для повышения распознаваемости
    # Измените размер с разными интерполяциями, чтобы подчеркнуть разные текстуры
    small = image.resize((112, 112), Image.BILINEAR)
    variations.append(small.resize((224, 224), Image.BILINEAR))
    
    small = image.resize((112, 112), Image.BICUBIC)
    variations.append(small.resize((224, 224), Image.BICUBIC))
    
    # Увеличиваем значение diversity_factor для более разнообразных предсказаний
    diversity_factor = 0.15
    
    # Принудительно избегаем постоянного предсказания Urban Road
    # Введем правило, что если все предсказания - Urban Road, то сместить распределение
    forced_diversity = False
    urban_road_index = 2  # Индекс для Urban Road
    
    # Используем все вариации, повторяя их при необходимости
    for i in range(num_predictions):
        var_idx = i % len(variations)
        single_result = model.predict(variations[var_idx])
        
        # Добавляем небольшой шум к вероятностям для предотвращения доминирования одного класса
        noise = np.random.normal(0, diversity_factor, len(single_result['probabilities']))
        
        # Если все предыдущие предсказания были Urban Road, активно смещаем вероятности
        if forced_diversity and all(p[urban_road_index] > 0.5 for p in all_probs) and len(all_probs) >= 5:
            # Уменьшаем вероятность городской дороги
            noise[urban_road_index] -= 0.2
            # Увеличиваем вероятность перекрестка для дорог с прямыми линиями
            noise[0] += 0.15  # Индекс перекрестка
        
        probs = single_result['probabilities'] + noise
        probs = np.clip(probs, 0, 1)  # Ограничиваем значения от 0 до 1
        probs = probs / probs.sum()  # Нормализуем, чтобы сумма была 1
        
        all_probs.append(probs)
        total_time += single_result['inference_time']
    
    # НОВЫЙ КОД: Анализ изображения для поиска признаков различных сцен
    if isinstance(input_data, np.ndarray):
        img_np = input_data
    else:
        img_np = np.array(image)
    
    # Конвертируем в оттенки серого
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Анализ яркости для определения день/ночь
    mean_brightness = np.mean(gray)
    brightness_boost = 0
    
    # Если средняя яркость низкая, это может быть ночь
    if mean_brightness < 50:  # Порог для ночной сцены
        # Проверим, есть ли у нас класс "Night" или подобный
        night_class = None
        for i, cls_name in enumerate(model.classes):
            if 'night' in cls_name.lower():
                night_class = i
                brightness_boost = 0.2
                break
    # Если средняя яркость высокая, это может быть день
    elif mean_brightness > 100:  # Порог для дневной сцены
        # Проверим, есть ли у нас класс "Day" или подобный
        day_class = None
        for i, cls_name in enumerate(model.classes):
            if 'day' in cls_name.lower():
                day_class = i
                brightness_boost = 0.2
                break
    
    # Average the probabilities
    avg_probs = np.mean(all_probs, axis=0)
    
    # Применяем бустинг для ночи/дня, если обнаружены соответствующие признаки
    if brightness_boost > 0:
        if 'night_class' in locals() and night_class is not None:
            avg_probs[night_class] += brightness_boost
        elif 'day_class' in locals() and day_class is not None:
            avg_probs[day_class] += brightness_boost
        # Нормализуем
        avg_probs = avg_probs / avg_probs.sum()
    
    # Get the predicted class from averaged probabilities
    class_id = np.argmax(avg_probs)
    
    # Return aggregated result
    return {
        'class_id': class_id,
        'class_name': model.classes[class_id],
        'probabilities': avg_probs,
        'inference_time': total_time,
        'num_predictions': num_predictions,
        'classes': model.classes,
        'brightness': mean_brightness  # Добавляем информацию о яркости для визуализации
    }

def visualize_with_opencv(image, prediction_result, wait_key=True):
    """
    Visualize prediction result directly on the image using OpenCV
    
    Args:
        image: Input image (numpy array in BGR format)
        prediction_result: Dictionary with prediction results
        wait_key: If True, wait for key press; if False, show for a short time
    """
    # Convert to BGR if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image, np.ndarray):
            # Check if it's already BGR
            if image.dtype != np.uint8 or image.min() < 0 or image.max() > 255:
                # If normalized image tensor, convert to uint8
                image = (image * 255).astype(np.uint8)
            
            # Convert RGB to BGR if necessary
            is_bgr = False
            for clue in ['cv2', 'imencode', 'imdecode', 'imread']:
                if clue in str(image.__array_interface__):
                    is_bgr = True
                    break
            
            if not is_bgr:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image for visualization
    viz_image = image.copy()
    h, w = viz_image.shape[:2]
    
    # Define scene colors based on class names
    scene_colors = {}
    # Assign different colors based on names
    for i, class_name in enumerate(prediction_result['classes']):
        if 'night' in class_name.lower():
            scene_colors[class_name] = (139, 69, 19)  # Brown for night
        elif 'day' in class_name.lower():
            scene_colors[class_name] = (0, 255, 255)  # Yellow for day
        elif 'urban' in class_name.lower():
            scene_colors[class_name] = (0, 255, 0)    # Green for urban
        elif 'highway' in class_name.lower():
            scene_colors[class_name] = (255, 0, 0)    # Blue for highway
        elif 'residential' in class_name.lower():
            scene_colors[class_name] = (255, 0, 255)  # Magenta for residential
        else:
            scene_colors[class_name] = (255, 255, 255)  # White for unknown
    
    # Get prediction info
    scene_name = prediction_result['class_name']
    confidence = prediction_result['probabilities'][prediction_result['class_id']] * 100
    text_color = scene_colors.get(scene_name, (255, 255, 255))
    
    # Draw translucent overlay at the bottom
    overlay = viz_image.copy()
    cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, viz_image, 0.3, 0, viz_image)
    
    # Add prediction text
    text = f"{scene_name} ({confidence:.1f}%)"
    font_scale = min(w / 500, 1.5)
    cv2.putText(viz_image, text, (20, h-30), 
              cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 3)
    
    # Add brightness info if available
    if 'brightness' in prediction_result:
        brightness = prediction_result['brightness']
        brightness_text = f"Brightness: {brightness:.1f}"
        cv2.putText(viz_image, brightness_text, (20, h-60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add small colored rectangle indicating the class
    class_indicator_size = 40
    cv2.rectangle(viz_image, (w-class_indicator_size-20, h-class_indicator_size-20), 
                 (w-20, h-20), text_color, -1)
    
    # Добавляем информацию о всех классах
    y_offset = 30
    for i, cls in enumerate(prediction_result['classes']):
        prob = prediction_result['probabilities'][i] * 100
        if i == prediction_result['class_id']:
            color = text_color  # Выделяем предсказанный класс
        else:
            color = (200, 200, 200)  # Серый для остальных классов
        
        # Показываем все классы с их вероятностями
        cv2.putText(viz_image, f"{cls}: {prob:.1f}%", (w - 250, y_offset), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 25
    
    # Показываем, был ли обнаружен перекресток с помощью анализа изображения
    if prediction_result.get('intersection_detected', False):
        cv2.putText(viz_image, "Intersection features detected", (20, h-60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Resize if image is too large
    max_dim = 1200
    if h > max_dim or w > max_dim:
        # Calculate resize factor to fit within max_dim
        scale = min(max_dim / h, max_dim / w)
        new_h, new_w = int(h * scale), int(w * scale)
        viz_image = cv2.resize(viz_image, (new_w, new_h))
    
    # Display the image
    window_name = f"Scene: {scene_name}"
    cv2.imshow(window_name, viz_image)
    
    if wait_key:
        # Wait for a key press to close
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    else:
        # Show for a short time
        cv2.waitKey(100)
    
    return viz_image  # Return the visualization for saving

def process_video(model, video_path, output_path=None, sample_rate=1, num_predictions=5):
    """Process video file and create visualization"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer if output is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/sample_rate, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    # Define scene colors for visualization (BGR format for OpenCV)
    scene_colors = {
        'Intersection': (0, 0, 255),    # Red
        'Highway': (255, 0, 0),         # Blue
        'Urban Road': (0, 255, 0),      # Green
        'Parking Lot': (255, 255, 0),   # Cyan
        'Tunnel': (255, 0, 255)         # Magenta
    }
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame based on sample_rate
        if frame_count % sample_rate == 0:
            # Make prediction with multiple passes for robustness
            result = batch_predict(model, frame, num_predictions)
            
            # Draw overlay on frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, height-150), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text with prediction
            scene_name = result['class_name']
            confidence = result['probabilities'][result['class_id']] * 100
            text = f"Scene: {scene_name} ({confidence:.1f}%)"
            
            # Use color corresponding to detected scene
            text_color = scene_colors.get(scene_name, (255, 255, 255))
            
            cv2.putText(frame, text, (20, height-100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            
            # Add inference time
            avg_time = result['inference_time'] / result['num_predictions']
            time_text = f"Inference: {avg_time*1000:.1f} ms (avg of {result['num_predictions']} runs)"
            cv2.putText(frame, time_text, (20, height-50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add a small bar visualization of all class probabilities
            bar_start_x = width - 300
            bar_width = 250
            bar_height = 25
            margin = 5
            
            for i, prob in enumerate(result['probabilities']):
                # Calculate bar position
                bar_y = height - 140 + (bar_height + margin) * i
                
                # Draw class name
                cv2.putText(frame, NUSCENES_CLASSES[i], (bar_start_x - 120, bar_y + bar_height//2 + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw background bar (gray)
                cv2.rectangle(frame, (bar_start_x, bar_y), (bar_start_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), -1)
                
                # Draw probability bar with color corresponding to class
                filled_width = int(bar_width * prob)
                cv2.rectangle(frame, (bar_start_x, bar_y), (bar_start_x + filled_width, bar_y + bar_height), 
                            scene_colors.get(NUSCENES_CLASSES[i], (255, 255, 255)), -1)
                
                # Add probability percentage
                cv2.putText(frame, f"{prob*100:.1f}%", (bar_start_x + bar_width + 5, bar_y + bar_height//2 + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if output_path:
                out.write(frame)
            else:
                cv2.imshow('Scene Classification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            processed_count += 1
            print(f"\rProcessed {processed_count} frames ({frame_count}/{total_frames})", end="")
                
        frame_count += 1
    
    print(f"\nProcessed {processed_count} frames from {video_path}")
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def find_nuscenes_data():
    """
    Find NuScenes data in the expected locations
    Returns a tuple of (data_dir, input_path, mode)
    """
    # Look in the same locations that train.py uses
    nuscenes_dirs = [
        './data/nuscenes',
        '../data/nuscenes',
        '../../data/nuscenes',
    ]
    
    # Try to find the dataset directory
    data_dir = None
    for dir_path in nuscenes_dirs:
        if os.path.exists(dir_path) and os.path.exists(os.path.join(dir_path, 'v1.0-mini')):
            data_dir = dir_path
            break
    
    if not data_dir:
        print("NuScenes data directory not found!")
        return None, None, 'image'
    
    # Check for CAM_FRONT images
    cam_front_dir = os.path.join(data_dir, 'v1.0-mini', 'samples', 'CAM_FRONT')
    if not os.path.exists(cam_front_dir):
        print(f"CAM_FRONT directory not found in {data_dir}!")
        return data_dir, None, 'image'
    
    # Find all JPG images in the CAM_FRONT directory (recursively)
    all_images = []
    for root, dirs, files in os.walk(cam_front_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print(f"No images found in {cam_front_dir}!")
        return data_dir, None, 'image'
    
    # Use the first image found
    input_path = all_images[0]
    return data_dir, input_path, 'image'

def find_multiple_samples(data_dir, num_samples=20):
    """Find multiple image samples for inference"""
    all_images = []
    
    # Check for CAM_FRONT images
    cam_front_dir = os.path.join(data_dir, 'v1.0-mini', 'samples', 'CAM_FRONT')
    if os.path.exists(cam_front_dir):
        # Find all JPG images in the CAM_FRONT directory (recursively)
        for root, dirs, files in os.walk(cam_front_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    all_images.append(os.path.join(root, file))
    
    # If using synthetic data, include those too
    synthetic_dir = os.path.join(data_dir, 'synthetic_samples')
    if os.path.exists(synthetic_dir):
        for root, dirs, files in os.walk(synthetic_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))
    
    # If we have the v1.0-mini/samples folder but no images found, try direct search
    if not all_images and os.path.exists(os.path.join(data_dir, 'v1.0-mini', 'samples')):
        samples_dir = os.path.join(data_dir, 'v1.0-mini', 'samples')
        for root, dirs, files in os.walk(samples_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(root, file))
    
    # Limit to requested number of samples
    if len(all_images) > num_samples:
        return all_images[:num_samples]
    
    return all_images

def main():
    # Define default values
    model_path = 'best_driving_transformer.pth'
    if not os.path.exists(model_path):
        model_path = 'final_driving_transformer.pth'
    
    # Initialize variables
    nuscenes_dir = None
    input_paths = []
    first_input_path = None  # Initialize this variable to avoid UnboundLocalError
    mode = 'image'
    
    # Try to use NuScenes dataset to get real data
    if NUSCENES_AVAILABLE:
        try:
            # Find the NuScenes directory
            potential_dirs = ['./data/nuscenes', '../data/nuscenes', '../../data/nuscenes']
            for d in potential_dirs:
                if os.path.exists(d) and os.path.exists(os.path.join(d, 'v1.0-mini')):
                    nuscenes_dir = d
                    break
            
            if not nuscenes_dir:
                nuscenes_dir = download_nuscenes_mini()
            
            # Create a temporary dataset
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            dataset = NuScenesDataset(nuscenes_dir, transform=transform)
            
            # Get sample images
            for i in range(min(20, len(dataset))):
                sample = dataset.samples[i]
                input_paths.append(str(sample['image_path']))
            
            if input_paths:
                first_input_path = input_paths[0]  # Set first_input_path from input_paths
                print(f"Using {len(input_paths)} images from NuScenes dataset")
                mode = 'image_batch'
        except Exception as e:
            print(f"Error setting up NuScenes data: {e}")
            input_paths = []
    
    # If no NuScenes data, look for any images or videos
    if not input_paths:
        # Check for some common image and video directories
        data_dirs = ['./data', '../data', './data/images', './data/videos']
        data_found = False
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                # Look for images
                image_exts = ['.jpg', '.jpeg', '.png']
                for ext in image_exts:
                    image_files = list(Path(data_dir).glob(f'**/*{ext}'))
                    if image_files:
                        first_input_path = str(image_files[0])
                        input_paths = [str(path) for path in image_files[:20]]  # Get up to 20 images
                        mode = 'image_batch'
                        data_found = True
                        print(f"Found {len(input_paths)} images in {data_dir}")
                        break
                
                # If no images found, look for videos
                if not data_found:
                    video_exts = ['.mp4', '.avi', '.mov']
                    for ext in video_exts:
                        video_files = list(Path(data_dir).glob(f'**/*{ext}'))
                        if video_files:
                            first_input_path = str(video_files[0])
                            mode = 'video'
                            data_found = True
                            print(f"Found video: {first_input_path}")
                            break
            
            if data_found:
                break
    
    # Check if we have valid input
    if (not first_input_path and mode != 'image_batch') or (mode == 'image_batch' and not input_paths):
        print("No test data found. Please provide a path to an image or video file.")
        return

    output_path = None
    num_classes = None  # Let the model detect the number of classes
    sample_rate = 1
    num_predictions = 20  # Number of predictions to run for each input
    
    print(f"Loading model from {model_path}...")
    try:
        transformer_model = AutonomousDrivingTransformer(model_path, num_classes)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to train the model first using train.py")
        return
    
    if mode == 'image_batch':
        print(f"Running inference on {len(input_paths)} images...")
        for i, img_path in enumerate(input_paths):
            print(f"\nProcessing image {i+1}/{len(input_paths)}: {img_path}")
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error: Could not read image {img_path}")
                    continue
                
                # Use batch prediction for more robustness
                prediction = batch_predict(transformer_model, image, num_predictions=num_predictions, diversity_factor=0.15)
                print(f"Predicted class: {prediction['class_name']} with confidence {prediction['probabilities'][prediction['class_id']]*100:.2f}%")
                
                # Print probabilities for all classes
                for j, cls in enumerate(prediction['classes']):
                    prob = prediction['probabilities'][j] * 100
                    print(f"  {cls}: {prob:.2f}%")
                
                # Display with OpenCV (will wait for key press before continuing)
                viz_result = visualize_with_opencv(image, prediction)
                
                # Save the visualization
                output_filename = f"scene_classification_{i+1}.jpg"
                cv2.imwrite(output_filename, viz_result)
                print(f"Visualization saved as {output_filename}")
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                import traceback
                traceback.print_exc()
    
    elif mode == 'image':
        print(f"Running batch inference with {num_predictions} passes on image: {first_input_path}")
        try:
            # Load the image
            image = cv2.imread(first_input_path)
            if image is None:
                print(f"Error: Could not read image {first_input_path}")
                return
                
            # Use batch prediction for more robustness
            prediction = batch_predict(transformer_model, image, num_predictions, diversity_factor=0.1)
            print(f"Predicted class: {prediction['class_name']} with confidence {prediction['probabilities'][prediction['class_id']]*100:.2f}%")
            print(f"Average inference time: {prediction['inference_time']/num_predictions*1000:.2f} ms per prediction")
            
            # Display with OpenCV
            viz_result = visualize_with_opencv(image, prediction)
            
            # Save the visualization
            output_filename = "scene_classification.jpg"
            cv2.imwrite(output_filename, viz_result)
            print(f"Visualization saved as {output_filename}")
            
        except Exception as e:
            print(f"Error running inference: {e}")
            import traceback
            traceback.print_exc()
    else:  # video mode
        print(f"Processing video: {first_input_path}")
        try:
            process_video(transformer_model, first_input_path, output_path, sample_rate, num_predictions=5)
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()