import gradio as gr
import json
import time
import os
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any
import logging
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simple imports without complex dependencies
try:
    from src.character_pipeline import create_pipeline
    PIPELINE_AVAILABLE = True
    print("✅ RL Pipeline loaded successfully!")
except Exception as e:
    print(f"⚠️ Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCharacterApp:
    def __init__(self):
        self.pipeline = None
        if PIPELINE_AVAILABLE:
            try:
                self.pipeline = create_pipeline({
                    'use_rl_primary': True,
                    'rl_model_path': None
                })
                logger.info("✅ RL Pipeline initialized successfully")
            except Exception as e:
                logger.error(f"❌ Pipeline initialization failed: {e}")
                self.pipeline = None
    
    def extract_attributes(self, image):
        if image is None:
            return "Please upload an image first.", "{}", "No image provided"
        
        try:
            start_time = time.time()
            
            if self.pipeline and PIPELINE_AVAILABLE:
                # Use real RL pipeline
                attributes = self.pipeline.extract_from_image(image)
                processing_time = time.time() - start_time
                
                # Format output
                formatted_output = "**🎭 Character Attributes Extracted:**\n\n"
                attr_dict = attributes.to_dict() if hasattr(attributes, 'to_dict') else {
                    "Age": getattr(attributes, 'age', 'Unknown'),
                    "Gender": getattr(attributes, 'gender', 'Unknown'),
                    "Hair Color": getattr(attributes, 'hair_color', 'Unknown'),
                    "Eye Color": getattr(attributes, 'eye_color', 'Unknown'),
                    "Confidence": getattr(attributes, 'confidence_score', 0.0)
                }
                
                for key, value in attr_dict.items():
                    if key == "Confidence" or "Score" in key:
                        formatted_output += f"**{key}:** {value:.3f}\n"
                    else:
                        formatted_output += f"**{key}:** {value}\n"
                
                json_output = json.dumps(attr_dict, indent=2)
                stats = f"⚡ Processing Time: {processing_time:.2f}s\n🤖 Mode: RL Pipeline\n✅ Status: Success"
                
            else:
                # Fallback mode with basic analysis
                processing_time = time.time() - start_time
                
                # Simple mock attributes
                attr_dict = {
                    "Age": "Young Adult",
                    "Gender": "Unknown",
                    "Hair Color": "Unknown", 
                    "Eye Color": "Unknown",
                    "Confidence": 0.5
                }
                
                formatted_output = "**🎭 Character Attributes (Fallback Mode):**\n\n"
                for key, value in attr_dict.items():
                    if key == "Confidence":
                        formatted_output += f"**{key}:** {value:.3f}\n"
                    else:
                        formatted_output += f"**{key}:** {value}\n"
                
                json_output = json.dumps(attr_dict, indent=2)
                stats = f"⚡ Processing Time: {processing_time:.2f}s\n🔄 Mode: Fallback\n⚠️ Status: Limited functionality"
            
            return formatted_output, json_output, stats
            
        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            logger.error(error_msg)
            
            error_dict = {
                "error": str(e),
                "status": "error"
            }
            return error_msg, json.dumps(error_dict, indent=2), "❌ Processing failed"

def create_interface():
    app = SimpleCharacterApp()
    
    with gr.Blocks(title="RL Character Extraction") as interface:
        gr.Markdown("""
        # 🎭 RL-Enhanced Character Attribute Extraction
        
        Upload a character image to extract detailed attributes using our RL-powered pipeline.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="📸 Upload Character Image"
                )
                
                extract_btn = gr.Button(
                    "🚀 Extract Attributes",
                    variant="primary"
                )
            
            with gr.Column():
                formatted_output = gr.Markdown(
                    label="📋 Extracted Attributes",
                    value="Upload an image and click 'Extract Attributes' to see results."
                )
                
                stats_output = gr.Textbox(
                    label="📊 Processing Stats",
                    lines=3
                )
        
        json_output = gr.Code(
            label="📄 JSON Output",
            language="json"
        )
        
        extract_btn.click(
            fn=app.extract_attributes,
            inputs=[image_input],
            outputs=[formatted_output, json_output, stats_output]
        )
    
    return interface

def main():
    logger.info("🚀 Starting Simple Character Attribute Extraction Interface...")
    
    interface = create_interface()
    port = int(os.environ.get("PORT", 7860))
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()