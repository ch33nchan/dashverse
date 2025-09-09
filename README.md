# Character Attribute Extraction Pipeline

A scalable system for extracting structured character attributes from anime/manga images using computer vision and reinforcement learning.

## Overview

This project implements a production-ready pipeline that automatically extracts character attributes (age, gender, hair color, clothing style, etc.) from character images. The system combines visual analysis using CLIP with text tag parsing and uses reinforcement learning to optimize the fusion of multiple extraction methods.

## Dataset and Training Environment

The system was developed and trained using:
- **Dataset**: Danbooru character images from the cagliostrolab/860k-ordered-tags collection
- **Training Environment**: MacBook with Apple Silicon
- **Sample Size**: 5,369 character images with corresponding text tags
- **Processing**: CPU-based inference with optimized batching

The dataset contains anime/manga character images paired with descriptive tags in Danbooru format (comma-separated attributes like "1girl, black hair, red eyes, school uniform").

## Technical Approach

### Multi-Modal Architecture

The pipeline uses three complementary extraction methods:

1. **Visual Analysis**: CLIP model (openai/clip-vit-base-patch32) for zero-shot image classification
2. **Text Processing**: Rule-based parser for Danbooru tag extraction
3. **Intelligent Fusion**: Reinforcement learning agent that learns optimal combination strategies

### Core Components

```
Input Loader → Tag Parser → CLIP Analyzer → RL Optimizer → Attribute Fusion → Database Storage
```

- **Input Loader**: Handles image and text file processing
- **Tag Parser**: Extracts structured attributes from text tags using keyword mapping
- **CLIP Analyzer**: Performs visual attribute classification with confidence scoring
- **RL Optimizer**: Deep Q-Network that learns the best fusion strategy for different scenarios
- **Attribute Fusion**: Combines results using confidence-weighted voting
- **Database Storage**: SQLite caching system for processed results

### Reinforcement Learning Component

The RL system learns to optimize attribute extraction by:
- **State Space**: CLIP confidences, tag confidences, and agreement features
- **Action Space**: 6 different fusion strategies (conservative, aggressive, tag-priority, etc.)
- **Reward Function**: Based on extraction completeness, confidence, and accuracy
- **Training**: Continuous learning from each processing result

## Extracted Attributes

The system extracts the following character attributes:
- Age (child, teen, young adult, middle-aged, elderly)
- Gender (male, female, non-binary)
- Hair Style (ponytail, twintails, bun, curly, straight, etc.)
- Hair Color (black, brown, blonde, red, blue, green, etc.)
- Hair Length (short, medium, long)
- Eye Color (brown, blue, green, red, purple, etc.)
- Body Type (slim, muscular, curvy, etc.)
- Clothing Style (casual, formal, traditional, school uniform, etc.)
- Facial Expression (happy, sad, serious, etc.)
- Accessories (glasses, hat, jewelry, etc.)

## Performance Results

### Current Scale Performance
- **Processing Speed**: 2-5 images per second on MacBook
- **Success Rate**: 85-95% successful attribute extraction
- **Memory Usage**: Under 4GB RAM during batch processing
- **Dataset Coverage**: Successfully processed 5,369 images
- **Average Attributes**: 6-8 attributes extracted per image

### Example Output

For a sample character image, the system outputs:
```json
{
  "Age": "young adult",
  "Gender": "female",
  "Hair Style": "twintails",
  "Hair Color": "black",
  "Hair Length": "medium",
  "Eye Color": "red",
  "Body Type": "short",
  "Dress": "casual",
  "Confidence Score": 0.31
}
```

## Scalability Design

### Current Implementation Scale
The system is designed to handle the current dataset size efficiently while maintaining quality:
- **Batch Processing**: Configurable batch sizes (8-32 items)
- **Result Caching**: SQLite database prevents reprocessing
- **Memory Management**: Streaming data loading to avoid memory overflow
- **Modular Architecture**: Each component can be independently scaled or replaced

### Future Scalability Preparation
The architecture is prepared for larger scale deployment:
- **Horizontal Scaling**: Components can be distributed across multiple machines
- **Database Migration**: Easy transition from SQLite to PostgreSQL for production
- **API Integration**: Modular design supports REST API deployment
- **Batch Size Optimization**: Configurable processing parameters for different hardware
- **Caching Strategy**: Intermediate result storage for complex processing pipelines

### Estimated Large Scale Performance
Based on current performance metrics:
- **1 Million Images**: Approximately 3-6 days processing time
- **5 Million Images**: Approximately 2-3 weeks processing time
- **Optimization Potential**: 5-10x speedup possible with distributed processing

## Installation and Usage

### Prerequisites
- Python 3.9+
- 8GB+ RAM recommended
- 10GB+ disk space

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface
python gradio_app.py

# Run command line demo
python demo.py

# Open Jupyter notebook
jupyter notebook character_extraction_demo.ipynb
```

### Web Interface
The Gradio web application provides an interactive interface:
- Upload character images
- View extracted attributes in real-time
- Download results in JSON format
- Access at http://localhost:7860

## File Structure

```
├── pipeline/                 # Core pipeline components
│   ├── base.py              # Base classes and data structures
│   ├── input_loader.py      # Dataset loading
│   ├── tag_parser.py        # Text tag processing
│   ├── clip_analyzer.py     # Visual analysis
│   ├── rl_optimizer.py      # Reinforcement learning
│   ├── attribute_fusion.py  # Result combination
│   └── database.py          # Storage and caching
├── character_pipeline.py     # Main pipeline orchestrator
├── gradio_app.py            # Web interface
├── demo.py                  # Command line demo
├── character_extraction_demo.ipynb  # Jupyter notebook
├── data/                    # Database and cache files
└── requirements.txt         # Python dependencies
```

## Technical Implementation Details

### Model Selection
- **CLIP Model**: openai/clip-vit-base-patch32 chosen for balance of accuracy and speed
- **No Additional Training**: Uses pre-trained models with zero-shot classification
- **Reinforcement Learning**: Custom DQN implementation for fusion optimization
- **Text Processing**: Rule-based approach for reliable tag parsing

### Quality Assurance
- **Confidence Scoring**: Each extraction includes confidence metrics
- **Error Handling**: Graceful degradation when components fail
- **Validation**: Input validation and output schema enforcement
- **Logging**: Comprehensive logging for debugging and monitoring

### Production Readiness
- **Modular Design**: Easy to extend with new models or attributes
- **Configuration Management**: Flexible configuration system
- **Database Integration**: Structured storage with indexing
- **API Ready**: Architecture supports REST API deployment

## Results and Validation

The system has been validated on real-world data:
- **Dataset**: 5,369 Danbooru character images
- **Attribute Coverage**: Successfully extracts 8+ different attribute types
- **Consistency**: Produces standardized output format
- **Reliability**: Handles edge cases and corrupted inputs gracefully

## Future Development

The current implementation provides a solid foundation for:
- **Additional Models**: Integration of BLIP2 or other vision-language models
- **Distributed Processing**: Ray or Dask integration for cluster deployment
- **Advanced Features**: Multi-character detection, style transfer normalization
- **Production Deployment**: Docker containerization and cloud deployment

## Contact

This project demonstrates a complete end-to-end solution for character attribute extraction, combining modern AI techniques with practical engineering considerations for real-world deployment.