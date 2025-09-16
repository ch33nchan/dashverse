import unittest
from PIL import Image
import numpy as np
from character_pipeline import create_pipeline
from pipeline.base import CharacterAttributes

class TestCharacterAttributes(unittest.TestCase):

    def test_all_attributes_present(self):
        """
        Tests that the to_dict method includes all attributes, even if they are None.
        """
        # Create a dummy image
        dummy_image = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))

        # Instantiate the pipeline
        pipeline = create_pipeline()

        # Extract attributes
        attributes = pipeline.extract_from_image(dummy_image)
        attributes_dict = attributes.to_dict()

        # Define expected keys
        expected_keys = [
            'Age', 'Gender', 'Ethnicity', 'Hair Style', 'Hair Color', 
            'Hair Length', 'Eye Color', 'Body Type', 'Dress', 
            'Facial Expression', 'Accessories', 'Scars', 'Tattoos',
            'Confidence Score', 'Source Tags'
        ]

        # Check that all expected keys are present in the dictionary
        missing_keys = [key for key in expected_keys if key not in attributes_dict]
        
        self.assertEqual(len(missing_keys), 0, f"Missing keys in attributes_dict: {missing_keys}")
        
        print("Success: All expected character attributes are present in the output.")

if __name__ == '__main__':
    unittest.main()