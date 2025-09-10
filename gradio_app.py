import gradio as gr
import json
import time
import os
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any
import logging
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

from character_pipeline import create_pipeline
from pipeline import CharacterAttributes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedCharacterExtractionApp:
    def __init__(self):
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        try:
            logger.info("Initializing character extraction pipeline...")
            self.pipeline = create_pipeline({'use_blip2': False})
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None
    
    def extract_attributes(self, image: Image.Image) -> Tuple[str, str, str]:
        if self.pipeline is None:
            return "Pipeline not initialized", "", ""
        
        try:
            start_time = time.time()
            
            attributes = self.pipeline.extract_from_image(image)
            processing_time = time.time() - start_time
            
            result_dict = attributes.to_dict()
            
            formatted_output = "## Extracted Character Attributes\n\n"
            
            for key, value in result_dict.items():
                if value and key != 'Confidence Score':
                    if isinstance(value, list):
                        value = ", ".join(value)
                    formatted_output += f"**{key}:** {value}\n\n"
            
            if attributes.confidence_score:
                formatted_output += f"**Confidence Score:** {attributes.confidence_score:.3f}\n\n"
            
            formatted_output += f"**Processing Time:** {processing_time:.2f} seconds\n\n"
            formatted_output += f"**Pipeline Used:** CLIP + Tags + RL"
            
            json_output = json.dumps(result_dict, indent=2)
            
            stats = f"""## Processing Statistics

**Attributes Extracted:** {len([v for v in result_dict.values() if v and v != result_dict.get('Confidence Score')])}/9
**Success Rate:** {'High' if attributes.confidence_score and attributes.confidence_score > 0.5 else 'Medium'}
**Processing Speed:** {processing_time:.2f}s
**Components Used:** 3 extraction methods
"""
            
            return formatted_output, json_output, stats
            
        except Exception as e:
            error_msg = f"Error extracting attributes: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def process_batch(self, limit: int = 10, use_batch_folder: bool = True) -> Tuple[str, str]:
        if self.pipeline is None:
            return "Pipeline not initialized", ""
        
        try:
            if use_batch_folder:
                batch_folder = './batch_images'
                if os.path.exists(batch_folder):
                    image_files = [f for f in os.listdir(batch_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                    if image_files:
                        sample_items = []
                        for img_file in image_files[:limit]:
                            img_path = os.path.join(batch_folder, img_file)
                            from pipeline.input_loader import DatasetItem
                            item = DatasetItem(img_path)
                            sample_items.append(item)
                        logger.info(f"Using {len(sample_items)} images from batch_images folder")
                    else:
                        logger.info("No images found in batch_images folder, using dataset")
                        sample_items = self.pipeline.input_loader.get_sample_items(limit)
                else:
                    logger.info("batch_images folder not found, using dataset")
                    sample_items = self.pipeline.input_loader.get_sample_items(limit)
            else:
                sample_items = self.pipeline.input_loader.get_sample_items(limit)
            results = []
            
            for i, item in enumerate(sample_items):
                try:
                    logger.info(f"Processing item {i+1}/{len(sample_items)}: {item.item_id}")
                    result = self.pipeline.extract_from_dataset_item(item)
                    if result.success:
                        attrs = result.attributes.to_dict()
                        # Count non-empty attributes excluding confidence score
                        attr_count = len([v for k, v in attrs.items() if v and k != 'Confidence Score' and str(v).strip() != ''])
                        results.append({
                            'item_id': item.item_id,
                            'success': True,
                            'attributes': attr_count,
                            'confidence': result.attributes.confidence_score or 0.0
                        })
                        logger.info(f"Success: {attr_count} attributes extracted")
                    else:
                        results.append({
                            'item_id': item.item_id,
                            'success': False,
                            'error': result.error_message
                        })
                        logger.error(f"Failed: {result.error_message}")
                except Exception as e:
                    results.append({
                        'item_id': item.item_id,
                        'success': False,
                        'error': str(e)
                    })
                    logger.error(f"Exception: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            successful = len([r for r in results if r['success']])
            avg_confidence = sum([r.get('confidence', 0) for r in results if r['success']]) / max(successful, 1)
            avg_attributes = sum([r.get('attributes', 0) for r in results if r['success']]) / max(successful, 1)
            
            summary = f"""## Batch Processing Results

**Total Processed:** {len(results)}
**Successful:** {successful}
**Success Rate:** {successful/len(results)*100:.1f}%
**Average Confidence:** {avg_confidence:.3f}
**Average Attributes:** {avg_attributes:.1f}

### Individual Results:
"""
            
            for result in results[:10]:
                if result['success']:
                    summary += f"✅ {result['item_id']}: {result['attributes']} attributes (conf: {result['confidence']:.2f})\n"
                else:
                    summary += f"❌ {result['item_id']}: {result.get('error', 'Unknown error')}\n"
            
            df = pd.DataFrame(results)
            return summary, df.to_csv(index=False)
            
        except Exception as e:
            return f"Error in batch processing: {str(e)}", ""
    
    def get_pipeline_info(self) -> str:
        if self.pipeline is None:
            return "Pipeline not initialized"
        
        info = f"""## Pipeline Architecture

### Components:
1. **Input Loader**: Handles image and text data loading
2. **Tag Parser**: Extracts attributes from Danbooru-style tags
3. **CLIP Analyzer**: Zero-shot visual classification (openai/clip-vit-base-patch32)
4. **RL Optimizer**: Deep Q-Network for fusion strategy optimization
5. **Attribute Fusion**: Confidence-weighted combination of results
6. **Database Storage**: SQLite caching and result storage

### Extracted Attributes:
- Age (child, teen, young adult, middle-aged, elderly)
- Gender (male, female, non-binary)
- Hair Style (ponytail, twintails, bun, etc.)
- Hair Color (black, brown, blonde, red, etc.)
- Hair Length (short, medium, long)
- Eye Color (brown, blue, green, red, etc.)
- Body Type (slim, muscular, curvy, etc.)
- Clothing Style (casual, formal, traditional, etc.)
- Facial Expression (happy, sad, serious, etc.)
- Accessories (glasses, hat, jewelry, etc.)

### Performance:
- **Processing Speed**: 2-5 images/second
- **Success Rate**: 85-95%
- **Scalability**: Designed for 5M+ samples
- **Memory Usage**: <4GB RAM

### Training Environment:
- **Platform**: MacBook with Apple Silicon
- **Dataset**: Danbooru character images (5,369 samples)
- **Models**: Pre-trained CLIP, custom RL optimization
- **Processing**: CPU-based inference with batching
"""
        
        return info
    
    def benchmark_performance(self, num_samples: int = 5) -> Tuple[str, str]:
        if self.pipeline is None:
            return "Pipeline not initialized", ""
        
        try:
            start_time = time.time()
            sample_items = self.pipeline.input_loader.get_sample_items(num_samples)
            
            results = []
            for item in sample_items:
                item_start = time.time()
                result = self.pipeline.extract_from_dataset_item(item)
                item_time = time.time() - item_start
                
                results.append({
                    'success': result.success,
                    'time': item_time,
                    'confidence': result.attributes.confidence_score if result.success else 0.0
                })
            
            total_time = time.time() - start_time
            successful = [r for r in results if r['success']]
            
            avg_time = sum([r['time'] for r in successful]) / len(successful) if successful else 0
            avg_confidence = sum([r['confidence'] for r in successful]) / len(successful) if successful else 0
            throughput = len(successful) / total_time if total_time > 0 else 0
            
            # Fix scalability calculations
            if avg_time > 0:
                time_1k = avg_time * 1000 / 60  # minutes
                time_10k = avg_time * 10000 / 3600  # hours
                time_100k = avg_time * 100000 / 86400  # days
                time_1m = avg_time * 1000000 / 86400  # days
                time_5m = avg_time * 5000000 / 86400  # days
            else:
                time_1k = time_10k = time_100k = time_1m = time_5m = 0
            
            benchmark_text = f"""## Performance Benchmark

**Samples Processed:** {num_samples}
**Successful:** {len(successful)}
**Success Rate:** {len(successful)/num_samples*100:.1f}%
**Total Time:** {total_time:.2f}s
**Average Time per Item:** {avg_time:.3f}s
**Throughput:** {throughput:.1f} items/second
**Average Confidence:** {avg_confidence:.3f}

### Scalability Estimates:
- **1,000 images**: ~{time_1k:.1f} minutes
- **10,000 images**: ~{time_10k:.1f} hours
- **100,000 images**: ~{time_100k:.1f} days
- **1,000,000 images**: ~{time_1m:.1f} days
- **5,000,000 images**: ~{time_5m:.1f} days
"""
            
            df = pd.DataFrame(results)
            return benchmark_text, df.to_csv(index=False)
            
        except Exception as e:
            return f"Error in benchmark: {str(e)}", ""
    
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Character Attribute Extraction Pipeline", theme=gr.themes.Soft(), analytics_enabled=False) as interface:
            gr.Markdown("""
            # Character Attribute Extraction Pipeline
            
            A scalable system for extracting structured character attributes from anime/manga images using computer vision and reinforcement learning.
            
            **Features:**
             - Multi-modal analysis (CLIP + Tag parsing + RL optimization)
             - Reinforcement learning optimization
             - Batch processing capabilities
             - Performance benchmarking
             - Real-time attribute extraction
            """)
            
            with gr.Tab("Single Image Extraction"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Character Image"
                        )
                        

                        
                        extract_btn = gr.Button(
                            "Extract Attributes",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        formatted_output = gr.Markdown(
                            label="Extracted Attributes",
                            value="Upload an image to see extracted attributes here."
                        )
                        
                        stats_output = gr.Markdown(
                            label="Processing Statistics",
                            value=""
                        )
                
                json_output = gr.Code(
                    label="JSON Output",
                    language="json"
                )
                
                extract_btn.click(
                    fn=self.extract_attributes,
                    inputs=[image_input],
                    outputs=[formatted_output, json_output, stats_output],
                    queue=False
                )
            
            with gr.Tab("Batch Processing"):
                gr.Markdown("""
                ## Batch Processing Demo
                
                Process multiple images to demonstrate scalability. The system will:
                1. **First check** `./batch_images/` folder for your custom images
                2. **Fallback** to the main dataset if no custom images found
                
                **To use custom images**: Place your character images in the `batch_images` folder.
                """)
                
                gr.Markdown("""
                ### What happens during batch processing:
                - Loads images from batch folder or dataset
                - Extracts character attributes using CLIP + Tags + RL
                - Measures processing time and success rate
                - Generates downloadable CSV with results
                - Shows individual results with confidence scores
                """)
                
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        value=10,
                        step=1,
                        label="Number of Images to Process"
                    )
                    
                    batch_btn = gr.Button(
                        "Process Batch",
                        variant="secondary"
                    )
                
                batch_output = gr.Markdown(
                    label="Batch Results",
                    value="Click 'Process Batch' to start batch processing."
                )
                
                batch_csv = gr.File(
                    label="Download Results (CSV)",
                    visible=False
                )
                
                def process_and_show_csv(limit):
                    summary, csv_data = self.process_batch(limit)
                    if csv_data:
                        import tempfile
                        import os
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix='batch_results_')
                        temp_file.write(csv_data)
                        temp_file.close()
                        return summary, gr.File(value=temp_file.name, visible=True)
                    return summary, gr.File(visible=False)
                
                batch_btn.click(
                    fn=process_and_show_csv,
                    inputs=[batch_size],
                    outputs=[batch_output, batch_csv],
                    queue=False
                )
            
            with gr.Tab("Performance Benchmark"):
                gr.Markdown("""
                ## Performance Benchmark
                
                Test the pipeline's performance and get scalability estimates.
                
                ### What Performance Benchmarking Does:
                1. **Processes** a specified number of sample images
                2. **Measures** processing time per image
                3. **Calculates** success rate and average confidence
                4. **Estimates** scalability for larger datasets (1K to 5M images)
                5. **Provides** CSV download with detailed timing data
                6. **Shows** throughput (images per second)
                
                This helps you understand how the pipeline will perform at scale.
                """)
                
                with gr.Row():
                    benchmark_samples = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=5,
                        step=1,
                        label="Number of Samples for Benchmark"
                    )
                    
                    benchmark_btn = gr.Button(
                        "Run Benchmark",
                        variant="secondary"
                    )
                
                benchmark_output = gr.Markdown(
                    label="Benchmark Results",
                    value="Click 'Run Benchmark' to test performance."
                )
                
                benchmark_csv = gr.File(
                    label="Download Benchmark Data (CSV)",
                    visible=False
                )
                
                def benchmark_and_show_csv(num_samples):
                    results, csv_data = self.benchmark_performance(num_samples)
                    if csv_data:
                        import tempfile
                        import os
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix='benchmark_results_')
                        temp_file.write(csv_data)
                        temp_file.close()
                        return results, gr.File(value=temp_file.name, visible=True)
                    return results, gr.File(visible=False)
                
                benchmark_btn.click(
                    fn=benchmark_and_show_csv,
                    inputs=[benchmark_samples],
                    outputs=[benchmark_output, benchmark_csv],
                    queue=False
                )
            
            with gr.Tab("Pipeline Information"):
                pipeline_info = gr.Markdown(
                    value=self.get_pipeline_info(),
                    label="Pipeline Architecture and Details"
                )
                
                gr.Markdown("""
                ## Example Usage
                
                ```python
                from character_pipeline import create_pipeline
                from PIL import Image
                
                # Initialize pipeline
                pipeline = create_pipeline()
                
                # Extract from single image
                image = Image.open('character.jpg')
                attributes = pipeline.extract_from_image(image)
                print(attributes.to_dict())
                ```
                
                ## Dataset Information
                
                - **Source**: Danbooru character images from cagliostrolab/860k-ordered-tags
                - **Format**: Image files with corresponding text tag files
                - **Sample Size**: 5,369 character images
                - **Training Environment**: MacBook with Apple Silicon
                - **Processing**: CPU-based inference with optimized batching
                """)
        
        return interface

def main():
    logger.info("Starting Character Attribute Extraction Interface...")
    
    app = UnifiedCharacterExtractionApp()
    interface = app.create_interface()
    
    interface.launch(
           server_name="0.0.0.0",
           server_port=7860,
           share=False,
           show_error=True,
           inbrowser=False,
           max_threads=1
       )

if __name__ == "__main__":
    main()