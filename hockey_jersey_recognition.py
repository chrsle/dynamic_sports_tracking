import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import torch
from PIL import Image

@dataclass
class JerseyOCRConfig:
    """Configuration for jersey number recognition."""
    
    # Method selection
    use_vlm: bool = True  # False for ResNet classifier
    
    # SmolVLM2 settings
    vlm_model: str = "HuggingFaceTB/SmolVLM2-Instruct"
    vlm_max_tokens: int = 10
    
    # ResNet settings
    resnet_weights_path: Optional[str] = None
    
    # Recognition settings
    min_jersey_number: int = 1
    max_jersey_number: int = 99
    
    # Stabilization settings
    stabilization_threshold: int = 4  # Consecutive matches to confirm
    sample_interval: int = 8  # Sample every N frames
    
    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

class SmolVLM2JerseyOCR:
    """
    Jersey number reader using SmolVLM2 vision-language model.
    
    SmolVLM2 is a small but capable VLM that can read text in
    challenging conditions like motion blur and partial occlusion.
    """
    
    def __init__(self, config: Optional[JerseyOCRConfig] = None):
        """
        Initialize the OCR model.
        
        Args:
            config: OCR configuration
        """
        self.config = config or JerseyOCRConfig()
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Load SmolVLM2 model."""
        if self._model is not None:
            return
        
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        print(f"Loading SmolVLM2: {self.config.vlm_model}")
        
        self._processor = AutoProcessor.from_pretrained(
            self.config.vlm_model
        )
        
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.vlm_model,
            torch_dtype=torch.float16 if self.config.device == 'cuda' else torch.float32
        ).to(self.config.device)
        
        print(f"✓ SmolVLM2 loaded on {self.config.device}")
    
    def read_number(self, crop: np.ndarray) -> Optional[int]:
        """
        Read jersey number from player crop.
        
        Args:
            crop: Player image crop (BGR)
            
        Returns:
            Jersey number or None if not readable
        """
        self._load_model()
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        # Prompt for jersey number
        prompt = "What is the jersey number visible in this image? Reply with just the number, or 'none' if no number is visible."
        
        # Process
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.vlm_max_tokens
            )
        
        # Decode
        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract number from response
        return self._extract_number(response)
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract jersey number from model response.
        
        Args:
            text: Model output text
            
        Returns:
            Jersey number or None
        """
        # Find all numbers in response
        numbers = re.findall(r'\d+', text)
        
        for num_str in numbers:
            num = int(num_str)
            if self.config.min_jersey_number <= num <= self.config.max_jersey_number:
                return num
        
        return None
    
    def read_batch(self, crops: List[np.ndarray]) -> List[Optional[int]]:
        """
        Read jersey numbers from multiple crops.
        
        Args:
            crops: List of player image crops
            
        Returns:
            List of jersey numbers (or None for each)
        """
        return [self.read_number(crop) for crop in crops]

class ResNetJerseyClassifier:
    """
    Jersey number classifier using ResNet50.
    
    Treats jersey recognition as a 100-class classification
    problem (numbers 0-99).
    """
    
    def __init__(self, config: Optional[JerseyOCRConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: OCR configuration
        """
        self.config = config or JerseyOCRConfig()
        self._model = None
    
    def _load_model(self):
        """Load ResNet model."""
        if self._model is not None:
            return
        
        import torchvision.models as models
        import torch.nn as nn
        
        # Create model
        self._model = models.resnet50(pretrained=True)
        self._model.fc = nn.Linear(self._model.fc.in_features, 100)
        
        # Load weights if provided
        if self.config.resnet_weights_path:
            self._model.load_state_dict(
                torch.load(self.config.resnet_weights_path)
            )
            print(f"✓ Loaded weights from {self.config.resnet_weights_path}")
        
        self._model = self._model.to(self.config.device)
        self._model.eval()
        
        print(f"✓ ResNet50 classifier loaded on {self.config.device}")
    
    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess image for ResNet."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return transform(rgb)
    
    def read_number(self, crop: np.ndarray) -> Optional[int]:
        """
        Classify jersey number from player crop.
        
        Args:
            crop: Player image crop (BGR)
            
        Returns:
            Predicted jersey number
        """
        self._load_model()
        
        # Preprocess
        tensor = self._preprocess(crop).unsqueeze(0).to(self.config.device)
        
        # Predict
        with torch.no_grad():
            outputs = self._model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        # Return number if confident enough
        number = predicted.item()
        conf = confidence.item()
        
        if conf > 0.3 and self.config.min_jersey_number <= number <= self.config.max_jersey_number:
            return number
        
        return None
    
    def read_batch(self, crops: List[np.ndarray]) -> List[Optional[int]]:
        """
        Read jersey numbers from multiple crops.
        
        Args:
            crops: List of player image crops
            
        Returns:
            List of jersey numbers
        """
        return [self.read_number(crop) for crop in crops]

class PlayerIdentityManager:
    """
    Manages player identity tracking with stabilization.
    
    Requires multiple consecutive matching predictions before
    confirming a player's jersey number to avoid false IDs.
    """
    
    def __init__(self, config: Optional[JerseyOCRConfig] = None):
        """
        Initialize identity manager.
        
        Args:
            config: OCR configuration
        """
        self.config = config or JerseyOCRConfig()
        
        # Track predictions per player
        self.predictions: Dict[int, List[int]] = defaultdict(list)
        
        # Confirmed identities
        self.confirmed: Dict[int, Dict] = {}
        
        # Roster for name lookup
        self.roster: Dict[str, Dict[int, str]] = {}
    
    def add_prediction(
        self,
        track_id: int,
        number: int,
        team_id: int = -1
    ) -> bool:
        """
        Add a jersey number prediction for a tracked player.
        
        Args:
            track_id: Player tracker ID
            number: Predicted jersey number
            team_id: Team assignment (optional)
            
        Returns:
            True if identity is now confirmed
        """
        # Skip if already confirmed
        if track_id in self.confirmed:
            return True
        
        # Add prediction
        self.predictions[track_id].append(number)
        
        # Check for stabilization
        recent = self.predictions[track_id][-self.config.stabilization_threshold:]
        
        if (len(recent) >= self.config.stabilization_threshold and
            len(set(recent)) == 1):
            # All recent predictions match - confirm identity
            self.confirmed[track_id] = {
                'number': recent[0],
                'team_id': team_id
            }
            return True
        
        return False
    
    def get_label(self, track_id: int) -> str:
        """
        Get display label for a tracked player.
        
        Args:
            track_id: Player tracker ID
            
        Returns:
            Display label string
        """
        if track_id in self.confirmed:
            info = self.confirmed[track_id]
            number = info['number']
            team_id = info['team_id']
            
            # Try to get player name from roster
            name = self.roster.get(str(team_id), {}).get(number)
            
            if name:
                return f"{name} #{number}"
            return f"#{number}"
        
        elif track_id in self.predictions and self.predictions[track_id]:
            # Return best guess with question mark
            counter = Counter(self.predictions[track_id])
            best = counter.most_common(1)[0][0]
            return f"#{best}?"
        
        return f"ID:{track_id}"
    
    def is_confirmed(self, track_id: int) -> bool:
        """Check if player identity is confirmed."""
        return track_id in self.confirmed
    
    def get_number(self, track_id: int) -> Optional[int]:
        """Get confirmed jersey number for player."""
        if track_id in self.confirmed:
            return self.confirmed[track_id]['number']
        return None
    
    def set_roster(self, roster: Dict[str, Dict[int, str]]):
        """
        Set team roster for name lookup.
        
        Args:
            roster: Dict mapping team_id -> {number: name}
        """
        self.roster = roster
    
    def reset(self):
        """Reset all identity tracking."""
        self.predictions.clear()
        self.confirmed.clear()

# Example usage
if __name__ == "__main__":
    # Initialize OCR
    config = JerseyOCRConfig(
        use_vlm=True,
        stabilization_threshold=4,
        sample_interval=8
    )
    
    if config.use_vlm:
        ocr = SmolVLM2JerseyOCR(config)
    else:
        ocr = ResNetJerseyClassifier(config)
    
    # Initialize identity manager
    identity_manager = PlayerIdentityManager(config)
    
    # Set example NHL roster
    roster = {
        "0": {  # Home team (e.g., Penguins)
            87: "Sidney Crosby",
            71: "Evgeni Malkin",
            58: "Kris Letang",
            59: "Jake Guentzel"
        },
        "1": {  # Away team (e.g., Oilers)
            97: "Connor McDavid",
            29: "Leon Draisaitl",
            93: "Ryan Nugent-Hopkins",
            18: "Zach Hyman"
        }
    }
    identity_manager.set_roster(roster)
    
    print("Jersey Recognition Module Ready!")
    print(f"\nConfiguration:")
    print(f"  - Method: {'SmolVLM2' if config.use_vlm else 'ResNet50'}")
    print(f"  - Stabilization threshold: {config.stabilization_threshold}")
    print(f"  - Sample interval: {config.sample_interval} frames")
    print(f"  - Device: {config.device}")
