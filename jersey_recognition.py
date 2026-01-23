"""
Unified jersey number recognition module.

Consolidates VLM-based and ResNet-based jersey OCR approaches
from hockey and basketball modules.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import torch

from shared_utils import BaseOCR, BaseConfig, get_device, bgr_to_rgb, bgr_to_pil


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JerseyRecognitionConfig(BaseConfig):
    """Configuration for jersey number recognition."""

    # Method selection
    method: str = 'vlm'  # 'vlm' or 'resnet'

    # VLM settings
    vlm_model: str = 'HuggingFaceTB/SmolVLM2-Instruct'
    vlm_max_tokens: int = 10

    # ResNet settings
    resnet_weights_path: Optional[str] = None

    # Number validation
    min_jersey_number: int = 0
    max_jersey_number: int = 99

    # Stabilization settings
    stabilization_threshold: int = 3
    sample_interval: int = 5


# =============================================================================
# Prediction Container
# =============================================================================

@dataclass
class JerseyPrediction:
    """Container for jersey number prediction."""
    number: Optional[int]
    confidence: float
    raw_output: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if prediction is valid."""
        return self.number is not None and 0 <= self.number <= 99


# =============================================================================
# VLM-based Jersey OCR
# =============================================================================

class VLMJerseyOCR(BaseOCR):
    """
    Jersey number OCR using vision-language models (SmolVLM2).

    VLMs can read jersey numbers even in challenging conditions
    like motion blur and partial occlusion.
    """

    PROMPT = "What is the jersey number in this image? Reply with just the number, or 'none' if not visible."

    def __init__(self, config: Optional[JerseyRecognitionConfig] = None):
        super().__init__()
        self.config = config or JerseyRecognitionConfig(method='vlm')
        self._processor = None

    def _load_model(self):
        """Load VLM model."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, AutoModelForVision2Seq

        self._processor = AutoProcessor.from_pretrained(self.config.vlm_model)

        dtype = torch.float16 if self.config.device == 'cuda' else torch.float32
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.vlm_model,
            torch_dtype=dtype
        ).to(self.config.device)

        print(f"Loaded VLM: {self.config.vlm_model}")

    def read(self, image: np.ndarray) -> Optional[str]:
        """Read jersey number from image."""
        prediction = self.predict(image)
        return str(prediction.number) if prediction.is_valid else None

    def predict(self, image: np.ndarray) -> JerseyPrediction:
        """
        Predict jersey number from image crop.

        Args:
            image: Player crop (BGR format)

        Returns:
            JerseyPrediction with number and confidence
        """
        self._load_model()

        pil_image = bgr_to_pil(image)

        inputs = self._processor(
            text=self.PROMPT,
            images=pil_image,
            return_tensors="pt"
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.vlm_max_tokens
            )

        response = self._processor.decode(outputs[0], skip_special_tokens=True)
        number = self._extract_number(response)

        return JerseyPrediction(
            number=number,
            confidence=0.8 if number is not None else 0.0,
            raw_output=response
        )

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract jersey number from model response."""
        numbers = re.findall(r'\d+', text)

        for num_str in numbers:
            num = int(num_str)
            if self.config.min_jersey_number <= num <= self.config.max_jersey_number:
                return num

        return None

    def predict_batch(self, images: List[np.ndarray]) -> List[JerseyPrediction]:
        """Predict jersey numbers for multiple images."""
        return [self.predict(img) for img in images]


# =============================================================================
# ResNet-based Jersey Classifier
# =============================================================================

class ResNetJerseyClassifier(BaseOCR):
    """
    Jersey number classifier using fine-tuned ResNet50.

    Treats jersey recognition as a 100-class classification problem (0-99).
    Faster than VLM but requires fine-tuning.
    """

    def __init__(self, config: Optional[JerseyRecognitionConfig] = None):
        super().__init__()
        self.config = config or JerseyRecognitionConfig(method='resnet')
        self._transform = None

    def _load_model(self):
        """Load ResNet model."""
        if self._model is not None:
            return

        import torchvision.models as models
        import torchvision.transforms as transforms

        self._model = models.resnet50(pretrained=True)
        self._model.fc = torch.nn.Linear(self._model.fc.in_features, 100)

        if self.config.resnet_weights_path:
            self._model.load_state_dict(
                torch.load(self.config.resnet_weights_path, map_location=self.config.device)
            )

        self._model = self._model.to(self.config.device)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Loaded ResNet50 classifier on {self.config.device}")

    def read(self, image: np.ndarray) -> Optional[str]:
        """Read jersey number from image."""
        prediction = self.predict(image)
        return str(prediction.number) if prediction.is_valid else None

    def predict(self, image: np.ndarray) -> JerseyPrediction:
        """
        Predict jersey number from image crop.

        Args:
            image: Player crop (BGR format)

        Returns:
            JerseyPrediction with number and confidence
        """
        self._load_model()

        rgb = bgr_to_rgb(image)
        tensor = self._transform(rgb).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        number = predicted.item()
        conf = confidence.item()

        if conf < 0.3:
            number = None

        return JerseyPrediction(
            number=number,
            confidence=conf,
            raw_output=f"Class {predicted.item()} with confidence {conf:.3f}"
        )

    def predict_batch(self, images: List[np.ndarray]) -> List[JerseyPrediction]:
        """Predict jersey numbers for multiple images with batching."""
        self._load_model()

        tensors = []
        for image in images:
            rgb = bgr_to_rgb(image)
            tensors.append(self._transform(rgb))

        batch = torch.stack(tensors).to(self.config.device)

        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)

        results = []
        for i in range(len(images)):
            number = predictions[i].item()
            conf = confidences[i].item()

            results.append(JerseyPrediction(
                number=number if conf >= 0.3 else None,
                confidence=conf,
                raw_output=f"Class {number} with confidence {conf:.3f}"
            ))

        return results


# =============================================================================
# Identity Manager with Stabilization
# =============================================================================

@dataclass
class PlayerIdentity:
    """Tracked player identity with prediction history."""
    track_id: int
    team_id: int = -1
    predictions: List[int] = field(default_factory=list)
    confirmed_number: Optional[int] = None
    player_name: Optional[str] = None

    def add_prediction(self, number: int, threshold: int = 3) -> bool:
        """
        Add prediction and check for confirmation.

        Uses stabilization: confirms after N consecutive matching predictions.

        Args:
            number: Predicted jersey number
            threshold: Consecutive matches needed

        Returns:
            True if number was confirmed
        """
        if self.confirmed_number is not None:
            return True

        self.predictions.append(number)

        if len(self.predictions) >= threshold:
            recent = self.predictions[-threshold:]
            if len(set(recent)) == 1:
                self.confirmed_number = recent[0]
                return True

        return False

    def get_best_guess(self) -> Optional[int]:
        """Get most common prediction if not confirmed."""
        if self.confirmed_number:
            return self.confirmed_number

        if not self.predictions:
            return None

        counter = Counter(self.predictions)
        return counter.most_common(1)[0][0]


class PlayerIdentityManager:
    """
    Manages player identities with stabilization across frames.

    Requires multiple consecutive matching predictions before
    confirming a player's jersey number.
    """

    def __init__(
        self,
        config: Optional[JerseyRecognitionConfig] = None,
        roster: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Initialize identity manager.

        Args:
            config: Recognition configuration
            roster: Dict mapping team_id -> {number: name}
        """
        self.config = config or JerseyRecognitionConfig()
        self.roster = roster or {}
        self.players: Dict[int, PlayerIdentity] = {}

    def add_prediction(
        self,
        track_id: int,
        prediction: JerseyPrediction,
        team_id: int = -1
    ) -> bool:
        """
        Add a jersey number prediction for a tracked player.

        Args:
            track_id: Player tracker ID
            prediction: Jersey prediction
            team_id: Team assignment

        Returns:
            True if identity is now confirmed
        """
        if not prediction.is_valid:
            return False

        if track_id not in self.players:
            self.players[track_id] = PlayerIdentity(
                track_id=track_id,
                team_id=team_id
            )

        player = self.players[track_id]
        confirmed = player.add_prediction(
            prediction.number,
            self.config.stabilization_threshold
        )

        if confirmed and player.confirmed_number is not None:
            player.player_name = self._lookup_name(team_id, player.confirmed_number)

        return confirmed

    def _lookup_name(self, team_id: int, number: int) -> Optional[str]:
        """Look up player name from roster."""
        team_roster = self.roster.get(str(team_id), {})
        return team_roster.get(number) or team_roster.get(str(number))

    def get_label(self, track_id: int) -> str:
        """Get display label for a player."""
        if track_id not in self.players:
            return f"ID:{track_id}"

        player = self.players[track_id]

        if player.player_name:
            return f"{player.player_name} #{player.confirmed_number}"
        elif player.confirmed_number is not None:
            return f"#{player.confirmed_number}"
        elif player.predictions:
            best = player.get_best_guess()
            return f"#{best}?" if best else f"ID:{track_id}"

        return f"ID:{track_id}"

    def is_confirmed(self, track_id: int) -> bool:
        """Check if player identity is confirmed."""
        if track_id not in self.players:
            return False
        return self.players[track_id].confirmed_number is not None

    def get_number(self, track_id: int) -> Optional[int]:
        """Get confirmed jersey number for player."""
        if track_id in self.players:
            return self.players[track_id].confirmed_number
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get identification statistics."""
        total = len(self.players)
        confirmed = sum(1 for p in self.players.values() if p.confirmed_number)
        named = sum(1 for p in self.players.values() if p.player_name)

        return {
            'total_players': total,
            'confirmed_numbers': confirmed,
            'named_players': named,
            'identification_rate': confirmed / total if total > 0 else 0
        }

    def reset(self):
        """Reset all identity tracking."""
        self.players.clear()


# =============================================================================
# Factory Function
# =============================================================================

def create_jersey_ocr(
    config: Optional[JerseyRecognitionConfig] = None
) -> BaseOCR:
    """
    Create jersey OCR based on configuration.

    Args:
        config: Recognition configuration

    Returns:
        Jersey OCR instance
    """
    config = config or JerseyRecognitionConfig()

    if config.method == 'vlm':
        return VLMJerseyOCR(config)
    else:
        return ResNetJerseyClassifier(config)


if __name__ == "__main__":
    # Example roster
    example_roster = {
        "0": {87: "Sidney Crosby", 71: "Evgeni Malkin"},
        "1": {97: "Connor McDavid", 29: "Leon Draisaitl"}
    }

    config = JerseyRecognitionConfig(
        method='vlm',
        stabilization_threshold=4
    )

    ocr = create_jersey_ocr(config)
    identity_manager = PlayerIdentityManager(config, example_roster)

    print("Jersey Recognition Module Ready!")
    print(f"Method: {config.method}")
    print(f"Stabilization threshold: {config.stabilization_threshold}")
