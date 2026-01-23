import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import torch
from PIL import Image

# Configuration
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM2-Instruct"
JERSEY_OCR_PROMPT = "What is the jersey number shown in this image? Reply with only the number, nothing else."

# Stabilization settings
STABILIZATION_THRESHOLD = 3  # Consecutive matching predictions needed
SAMPLING_INTERVAL = 5  # Sample every Nth frame for jersey number recognition

@dataclass
class JerseyPrediction:
    """Container for jersey number prediction results."""
    number: Optional[str]
    confidence: float
    raw_output: str
    
    @property
    def is_valid(self) -> bool:
        """Check if prediction is a valid jersey number."""
        if self.number is None:
            return False
        try:
            num = int(self.number)
            return 0 <= num <= 99
        except ValueError:
            return False

@dataclass
class PlayerIdentity:
    """Tracked player identity with jersey number history."""
    track_id: int
    team_id: int = -1
    predictions: List[str] = field(default_factory=list)
    confirmed_number: Optional[str] = None
    player_name: Optional[str] = None
    
    def add_prediction(self, number: str) -> bool:
        """
        Add a jersey number prediction and check for confirmation.
        
        Uses stabilization heuristic: confirms number after
        STABILIZATION_THRESHOLD consecutive matching predictions.
        
        Args:
            number: Predicted jersey number
            
        Returns:
            True if number was confirmed, False otherwise
        """
        if self.confirmed_number is not None:
            return True  # Already confirmed
        
        self.predictions.append(number)
        
        # Check last N predictions for stabilization
        if len(self.predictions) >= STABILIZATION_THRESHOLD:
            recent = self.predictions[-STABILIZATION_THRESHOLD:]
            if len(set(recent)) == 1:
                self.confirmed_number = recent[0]
                return True
        
        return False
    
    def get_best_prediction(self) -> Optional[str]:
        """
        Get the most common prediction if not yet confirmed.
        
        Returns:
            Confirmed number, most common prediction, or None
        """
        if self.confirmed_number:
            return self.confirmed_number
        
        if not self.predictions:
            return None
        
        counter = Counter(self.predictions)
        return counter.most_common(1)[0][0]

class SmolVLMJerseyOCR:
    """
    Jersey number OCR using SmolVLM2 vision-language model.
    
    SmolVLM2 is a compact VLM specialized for image-to-text tasks.
    Pre-trained mainly on document OCR, it transfers well to jersey numbers.
    """
    
    def __init__(
        self,
        model_name: str = SMOLVLM_MODEL,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize SmolVLM2 for jersey OCR.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
            except ImportError:
                raise ImportError(
                    "Please install transformers: pip install transformers"
                )
            
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
            ).to(self.device)
    
    def predict(self, image: np.ndarray) -> JerseyPrediction:
        """
        Predict jersey number from image crop.
        
        Args:
            image: Jersey number crop (BGR format)
            
        Returns:
            JerseyPrediction with number and confidence
        """
        self._load_model()
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": JERSEY_OCR_PROMPT}
                ]
            }
        ]
        
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        # Decode response
        response = self._processor.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Extract number from response
        number = self._extract_number(response)
        
        return JerseyPrediction(
            number=number,
            confidence=0.8 if number else 0.0,  # Simple confidence heuristic
            raw_output=response
        )
    
    def _extract_number(self, text: str) -> Optional[str]:
        """
        Extract jersey number from model response.
        
        Args:
            text: Raw model output
            
        Returns:
            Extracted number string or None
        """
        import re
        
        # Look for 1-2 digit numbers
        matches = re.findall(r'\b(\d{1,2})\b', text)
        
        if matches:
            # Return first valid jersey number (0-99)
            for match in matches:
                num = int(match)
                if 0 <= num <= 99:
                    return match
        
        return None
    
    def predict_batch(self, images: List[np.ndarray]) -> List[JerseyPrediction]:
        """
        Predict jersey numbers for multiple images.
        
        Args:
            images: List of jersey crop images
            
        Returns:
            List of JerseyPrediction objects
        """
        return [self.predict(img) for img in images]

class ResNetJerseyClassifier:
    """
    Jersey number classifier using fine-tuned ResNet.
    
    Alternative to VLM-based OCR, this approach treats jersey
    number recognition as a 100-class classification problem (0-99).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize ResNet jersey classifier.
        
        Args:
            model_path: Path to fine-tuned model weights (None for pretrained)
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        self._model = None
        self._transform = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                import torchvision.models as models
                import torchvision.transforms as transforms
            except ImportError:
                raise ImportError(
                    "Please install torchvision: pip install torchvision"
                )
            
            # Create ResNet model with 100 output classes
            self._model = models.resnet50(pretrained=True)
            self._model.fc = torch.nn.Linear(
                self._model.fc.in_features, 100
            )
            
            if self.model_path:
                self._model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
            
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Image transforms
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def predict(self, image: np.ndarray) -> JerseyPrediction:
        """
        Predict jersey number from image crop.
        
        Args:
            image: Jersey crop image (BGR format)
            
        Returns:
            JerseyPrediction with number and confidence
        """
        self._load_model()
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        tensor = self._transform(rgb_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        number = str(predicted.item())
        
        return JerseyPrediction(
            number=number,
            confidence=confidence.item(),
            raw_output=f"Class {number} with confidence {confidence.item():.3f}"
        )
    
    def predict_batch(self, images: List[np.ndarray]) -> List[JerseyPrediction]:
        """
        Predict jersey numbers for multiple images.
        
        Args:
            images: List of jersey crop images
            
        Returns:
            List of JerseyPrediction objects
        """
        self._load_model()
        
        predictions = []
        
        # Process in batch
        tensors = []
        for image in images:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb_image)
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted_classes = torch.max(probs, 1)
        
        for i in range(len(images)):
            number = str(predicted_classes[i].item())
            confidence = confidences[i].item()
            
            predictions.append(JerseyPrediction(
                number=number,
                confidence=confidence,
                raw_output=f"Class {number} with confidence {confidence:.3f}"
            ))
        
        return predictions

class PlayerIdentityManager:
    """
    Manages player identities across video frames.
    
    Tracks jersey number predictions per player, applies stabilization
    heuristic, and maps confirmed numbers to player names.
    """
    
    def __init__(
        self,
        roster: Optional[Dict[str, Dict[str, str]]] = None,
        valid_numbers: Optional[List[str]] = None
    ):
        """
        Initialize the identity manager.
        
        Args:
            roster: Dictionary mapping team_id to {jersey_number: player_name}
            valid_numbers: List of valid jersey numbers on court (constrains predictions)
        """
        self.roster = roster or {}
        self.valid_numbers = set(valid_numbers) if valid_numbers else None
        self.players: Dict[int, PlayerIdentity] = {}
    
    def get_or_create_player(self, track_id: int, team_id: int = -1) -> PlayerIdentity:
        """
        Get existing player identity or create new one.
        
        Args:
            track_id: Unique tracking ID for the player
            team_id: Team ID from team classifier
            
        Returns:
            PlayerIdentity object
        """
        if track_id not in self.players:
            self.players[track_id] = PlayerIdentity(
                track_id=track_id,
                team_id=team_id
            )
        
        return self.players[track_id]
    
    def add_prediction(
        self,
        track_id: int,
        prediction: JerseyPrediction,
        team_id: int = -1
    ) -> bool:
        """
        Add a jersey number prediction for a player.
        
        Args:
            track_id: Player tracking ID
            prediction: Jersey number prediction
            team_id: Team ID from classifier
            
        Returns:
            True if jersey number was confirmed
        """
        if not prediction.is_valid:
            return False
        
        number = prediction.number
        
        # Constrain to valid numbers if provided
        if self.valid_numbers and number not in self.valid_numbers:
            return False
        
        player = self.get_or_create_player(track_id, team_id)
        confirmed = player.add_prediction(number)
        
        # Look up player name if confirmed
        if confirmed and player.confirmed_number:
            player.player_name = self._lookup_player_name(
                team_id, player.confirmed_number
            )
        
        return confirmed
    
    def _lookup_player_name(
        self,
        team_id: int,
        jersey_number: str
    ) -> Optional[str]:
        """
        Look up player name from roster.
        
        Args:
            team_id: Team ID
            jersey_number: Confirmed jersey number
            
        Returns:
            Player name or None
        """
        team_roster = self.roster.get(str(team_id), {})
        return team_roster.get(jersey_number)
    
    def get_player_label(self, track_id: int) -> str:
        """
        Get display label for a player.
        
        Args:
            track_id: Player tracking ID
            
        Returns:
            Label string (name, number, or track ID)
        """
        if track_id not in self.players:
            return f"Player {track_id}"
        
        player = self.players[track_id]
        
        if player.player_name:
            return f"{player.player_name} #{player.confirmed_number}"
        elif player.confirmed_number:
            return f"#{player.confirmed_number}"
        elif player.predictions:
            best = player.get_best_prediction()
            return f"#{best}?" if best else f"Player {track_id}"
        else:
            return f"Player {track_id}"
    
    def get_confirmed_players(self) -> List[PlayerIdentity]:
        """
        Get all players with confirmed jersey numbers.
        
        Returns:
            List of PlayerIdentity objects with confirmed numbers
        """
        return [
            player for player in self.players.values()
            if player.confirmed_number is not None
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about player identification.
        
        Returns:
            Dictionary with identification statistics
        """
        total = len(self.players)
        confirmed = len(self.get_confirmed_players())
        named = len([p for p in self.players.values() if p.player_name])
        
        return {
            'total_players': total,
            'confirmed_numbers': confirmed,
            'named_players': named,
            'identification_rate': confirmed / total if total > 0 else 0
        }

# Example NBA roster format
EXAMPLE_ROSTER = {
    "0": {  # Team 0
        "23": "LeBron James",
        "3": "Anthony Davis",
        "15": "Austin Reaves",
        "28": "Rui Hachimura",
        "12": "Max Christie"
    },
    "1": {  # Team 1
        "30": "Stephen Curry",
        "11": "Klay Thompson",
        "23": "Draymond Green",
        "22": "Andrew Wiggins",
        "0": "Gary Payton II"
    }
}

# Example usage
if __name__ == "__main__":
    # Initialize OCR model
    # ocr = SmolVLMJerseyOCR()
    
    # Or use ResNet classifier
    # classifier = ResNetJerseyClassifier(model_path="jersey_classifier.pth")
    
    # Initialize identity manager with roster
    identity_manager = PlayerIdentityManager(
        roster=EXAMPLE_ROSTER,
        valid_numbers=["23", "3", "15", "28", "12", "30", "11", "22", "0"]
    )
    
    # Example: Process jersey predictions
    # for frame in video:
    #     # Get jersey crops from detections
    #     # Predict numbers
    #     # Add predictions to identity manager
    #     pass
    
    print("Jersey recognition module initialized!")
    print(f"Sample roster: {EXAMPLE_ROSTER}")
