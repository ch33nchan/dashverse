"""Input loader stage for the character attribute extraction pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from PIL import Image
import logging
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from datasets import Dataset as HFDataset, load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFDataset = None

from .base import PipelineStage

logger = logging.getLogger(__name__)

class DatasetItem:
    """Represents a single item from the dataset."""
    
    def __init__(self, image_path: str, text_path: Optional[str] = None, tags: Optional[str] = None):
        self.image_path = image_path
        self.text_path = text_path
        self.tags = tags
        self.item_id = Path(image_path).stem
    
    def load_image(self) -> Image.Image:
        """Load and return the PIL Image."""
        try:
            return Image.open(self.image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {self.image_path}: {e}")
            raise
    
    def load_tags(self) -> str:
        """Load tags from text file or return provided tags."""
        if self.tags:
            return self.tags
        
        if self.text_path and os.path.exists(self.text_path):
            try:
                with open(self.text_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to load tags from {self.text_path}: {e}")
                return ""
        
        return ""

class CharacterDataset(Dataset):
    """PyTorch Dataset for character images and attributes."""
    
    def __init__(self, items: List[DatasetItem], transform=None):
        self.items = items
        self.transform = transform
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        image = item.load_image()
        tags = item.load_tags()
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'tags': tags,
            'item_id': item.item_id,
            'image_path': item.image_path
        }

class InputLoader(PipelineStage):
    """Loads images and associated text data from the dataset."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("InputLoader", config)
        if config:
            self.dataset_path = config.get('dataset_path', './continued/sensitive')
            self.batch_size = config.get('batch_size', 32)
            self.num_workers = config.get('num_workers', 4)
        else:
            self.dataset_path = './continued/sensitive'
            self.batch_size = 32
            self.num_workers = 4
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def discover_dataset_items(self) -> List[DatasetItem]:
        """Discover all image-text pairs in the dataset directory."""
        dataset_path = Path(self.dataset_path)
        items = []
        
        if not dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            return items
        
        # Find all image files
        image_files = []
        for ext in self.supported_image_formats:
            image_files.extend(dataset_path.glob(f"*{ext}"))
        
        self.logger.info(f"Found {len(image_files)} image files")
        
        for image_path in image_files:
            # Look for corresponding text file
            text_path = image_path.with_suffix('.txt')
            
            item = DatasetItem(
                image_path=str(image_path),
                text_path=str(text_path) if text_path.exists() else None
            )
            items.append(item)
        
        self.logger.info(f"Created {len(items)} dataset items")
        return items
    
    def load_single_item(self, item_path: Union[str, Path]) -> DatasetItem:
        """Load a single item by path."""
        item_path = Path(item_path)
        
        if not item_path.exists():
            raise FileNotFoundError(f"Item not found: {item_path}")
        
        # If it's an image, look for corresponding text
        if item_path.suffix.lower() in self.supported_image_formats:
            text_path = item_path.with_suffix('.txt')
            return DatasetItem(
                image_path=str(item_path),
                text_path=str(text_path) if text_path.exists() else None
            )
        
        # If it's a text file, look for corresponding image
        elif item_path.suffix.lower() == '.txt':
            for ext in self.supported_image_formats:
                image_path = item_path.with_suffix(ext)
                if image_path.exists():
                    return DatasetItem(
                        image_path=str(image_path),
                        text_path=str(item_path)
                    )
            
            # No corresponding image found, create text-only item
            with open(item_path, 'r', encoding='utf-8') as f:
                tags = f.read().strip()
            
            return DatasetItem(
                image_path=None,
                text_path=str(item_path),
                tags=tags
            )
        
        else:
            raise ValueError(f"Unsupported file format: {item_path.suffix}")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return loaded content."""
        if isinstance(input_data, (str, Path)):
            # Single item path provided
            item = self.load_single_item(input_data)
        elif isinstance(input_data, DatasetItem):
            # DatasetItem provided directly
            item = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        result = {
            'item_id': item.item_id,
            'image': None,
            'tags': '',
            'image_path': item.image_path,
            'text_path': item.text_path
        }
        
        # Load image if available
        if item.image_path and os.path.exists(item.image_path):
            try:
                result['image'] = item.load_image()
                self.logger.debug(f"Loaded image: {item.image_path}")
            except Exception as e:
                self.logger.error(f"Failed to load image {item.image_path}: {e}")
                result['image'] = None
        
        # Load tags if available
        try:
            result['tags'] = item.load_tags()
            self.logger.debug(f"Loaded tags: {len(result['tags'])} characters")
        except Exception as e:
            self.logger.error(f"Failed to load tags: {e}")
            result['tags'] = ''
        
        return result
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate that input can be processed."""
        if isinstance(input_data, (str, Path)):
            return Path(input_data).exists()
        elif isinstance(input_data, DatasetItem):
            return True
        return False
    
    def get_sample_items(self, n: int = 10) -> List[DatasetItem]:
        """Get a sample of dataset items for testing."""
        all_items = self.discover_dataset_items()
        return all_items[:n] if len(all_items) >= n else all_items
    
    def create_pytorch_dataset(self, items: Optional[List[DatasetItem]] = None, transform=None) -> CharacterDataset:
        """Create a PyTorch Dataset from dataset items."""
        if items is None:
            items = self.discover_dataset_items()
        return CharacterDataset(items, transform=transform)
    
    def create_dataloader(self, items: Optional[List[DatasetItem]] = None, transform=None, 
                         batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader for batch processing."""
        dataset = self.create_pytorch_dataset(items, transform)
        batch_size = batch_size or self.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        images = [item['image'] for item in batch]
        tags = [item['tags'] for item in batch]
        item_ids = [item['item_id'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        
        return {
            'images': images,
            'tags': tags,
            'item_ids': item_ids,
            'image_paths': image_paths
        }
    
    def create_huggingface_dataset(self, items: Optional[List[DatasetItem]] = None) -> Optional[HFDataset]:
        """Create a HuggingFace Dataset for efficient batch processing."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available. Install with: pip install datasets")
            return None
        
        if items is None:
            items = self.discover_dataset_items()
        
        # Prepare data for HuggingFace Dataset
        data = {
            'image_path': [item.image_path for item in items],
            'tags': [item.load_tags() for item in items],
            'item_id': [item.item_id for item in items]
        }
        
        return HFDataset.from_dict(data)
    
    def process_with_hf_map(self, processing_fn, items: Optional[List[DatasetItem]] = None, 
                           batch_size: Optional[int] = None, num_proc: int = 4) -> Optional[HFDataset]:
        """Process dataset using HuggingFace datasets.map() for efficient batch inference."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available")
            return None
        
        dataset = self.create_huggingface_dataset(items)
        if dataset is None:
            return None
        
        batch_size = batch_size or self.batch_size
        
        # Apply processing function using datasets.map()
        processed_dataset = dataset.map(
            processing_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=['image_path'],  # Remove to save memory
            desc="Processing character attributes"
        )
        
        return processed_dataset
    
    def batch_process_iterator(self, items: Optional[List[DatasetItem]] = None, 
                              batch_size: Optional[int] = None) -> Iterator[List[DatasetItem]]:
        """Create an iterator for batch processing without loading all data into memory."""
        if items is None:
            items = self.discover_dataset_items()
        
        batch_size = batch_size or self.batch_size
        
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]