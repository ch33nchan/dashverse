"""Simple Gradio web interface for character attribute extraction."""

import gradio as gr
import json
import time
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any
import logging

from character_pipeline import create_pipeline
from pipeline import CharacterAttributes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGradioInterface:
    """Simple Gradio interface for character attribute extraction."""
    
    def __init__(self):
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the extraction pipeline."""
        try:
            logger.info("Initializing character extraction pipeline...")
            self.pipeline = create_pipeline()
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None
    
    def extract_attributes(self, image: Image.Image) -> Tuple[str, str]:
        """Extract character attributes from uploaded image."""
        if self.pipeline is None:
            return "Pipeline not initialized", ""
        
        try:
            start_time = time.time()
            
            # Extract attributes
            attributes = self.pipeline.extract_from_image(image)
            processing_time = time.time() - start_time
            
            # Format results
            result_dict = attributes.to_dict()
            
            # Create formatted output
            formatted_output = "## Extracted Character Attributes\n\n"
            
            for key, value in result_dict.items():
                if value and key != 'Confidence Score':
                    if isinstance(value, list):
                        value = ", ".join(value)
                    formatted_output += f"**{key}:** {value}\n\n"
            
            if attributes.confidence_score:
                formatted_output += f"**Confidence Score:** {attributes.confidence_score:.3f}\n\n"
            
            formatted_output += f"**Processing Time:** {processing_time:.2f} seconds"
            
            # Create JSON output
            json_output = json.dumps(result_dict, indent=2)
            
            return formatted_output, json_output
            
        except Exception as e:
            error_msg = f"Error extracting attributes: {str(e)}"
            logger.error(error_msg)
            return error_msg, ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Character Attribute Extraction") as interface:
            gr.Markdown("""
            # 🎭 Character Attribute Extraction Pipeline
            
            Upload an image to extract character attributes using CLIP + Reinforcement Learning.
            
            **Features:**
            - CLIP Visual Analysis (openai/clip-vit-base-patch32)
            - Danbooru Tag Parser
            - Reinforcement Learning Optimization
            - Scalable to 5M+ samples
            """)
            
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
            
            json_output = gr.Code(
                label="JSON Output",
                language="json"
            )
            
            extract_btn.click(
                fn=self.extract_attributes,
                inputs=[image_input],
                outputs=[formatted_output, json_output]
            )
            
            # Add example
            gr.Markdown("""
            ## Example
            
            Try uploading the demo image: `danbooru_1380555_f9c05b66378137705fb63e010d6259d8.png`
            
            Expected output:
            - Age: young adult
            - Gender: female  
            - Hair Style: twintails
            - Hair Color: black
            - Eye Color: red
            """)
        
        return interface

def main():
    """Main function to launch the Gradio interface."""
    logger.info("Starting Character Attribute Extraction Interface...")
    
    # Create interface
    app = SimpleGradioInterface()
    interface = app.create_interface()
    
    # Launch interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()