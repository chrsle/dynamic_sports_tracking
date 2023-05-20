# NHL Player and Goal Recognition System

## Project Description
This project uses a combination of Machine Learning (ML) and Computer Vision techniques to perform player recognition and tracking in National Hockey League (NHL) games, as well as goal recognition. The system is capable of analyzing game footage, recognizing and tracking player movements, and identifying when a goal is scored. The goal of this project is to automate the process of game analysis and player performance assessment, enabling faster and more accurate insights into game dynamics.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
The project requires Python 3.6 or later and pip (Python package installer).

### Installation
1. Clone this repository to your local machine using `git clone https://github.com/yourusername/your-repository.git`
2. Navigate to the project directory with `cd your-repository`
3. Install the necessary Python packages by running the command: `pip install -r requirements.txt`

## Project Structure
This project is organized into several Python scripts, each with a specific role:

- `download_footage.py`: This script downloads NHL game footage from specified sources for use in model training and testing.
- `extract_frames.py`: This script extracts individual frames from the downloaded video files. These frames are used as input data for model training.
- `label_frames.py`: This script provides a utility for manually labeling players in the extracted frames. These labeled frames are used as training data for the player detection model.
- `data_augmentation.py`: This script applies a series of data augmentation techniques to the training data to create a more diverse set of examples and enhance model generalizability.
- `train_detection_model.py`: This script trains the player detection model using the labeled and augmented training data.
- `detection_model_utils.py`: This script contains utility functions used in training and applying the detection model, such as model saving/loading and image preprocessing.
- `recognize_goals.py`: This script uses the output of the player detection model to recognize when a goal is scored based on player positions and movements.
- `goal_recognition_utils.py`: This script contains utility functions used in goal recognition, such as player trajectory analysis and goal event detection.
- `evaluation.py`: This script contains functions for evaluating the performance of the player detection and goal recognition models. It computes metrics like precision, recall, and F1 score, and generates plots to visualize model performance.
- `visualization.py`: This script contains functions for visualizing the results of player detection, player tracking, and goal recognition, such as displaying the input video with overlaid bounding boxes and player tracks, and plotting player movements and goal events over time.
- `train.py`: This is the main script for model training. It incorporates all the necessary steps to preprocess the data, train the models, and evaluate their performance. Trained models are saved to disk for later use.
- `infer.py`: This is the main script for making predictions with the trained models. It loads the models from disk and applies them to new, unseen video footage. The results of the inference process, such as annotated video frames or tracking data, can be saved to disk for further analysis.

## Usage
To train the models, navigate to the project directory and run the following command:

python train.py

To perform inference on new game footage, run the following command:

python infer.py --input_video path_to_your_video

Note: Replace `path_to_your_video` with the actual path to your input video file.

## Contributing
We appreciate all contributions to improve the project. Please feel free to submit pull requests or create issues to discuss any changes you wish to make. Here's a quick guide on how you can contribute:

1. Fork the project on GitHub.
2. Clone your fork to your local machine.
3. Create a new branch to work on your feature or bug fix.
4. Make your changes and test them thoroughly.
5. Commit your changes with clear, descriptive commit messages.
6. Push your changes to your fork on GitHub.
7. Submit a pull request to the main repository.

Before submitting a pull request, please ensure that your code adheres to our coding standards and that all tests pass.

## Acknowledgments
We would like to thank all contributors and users of this project. Your support is what makes this project so great. We are constantly amazed by the incredible work being done by the community and are always striving to make this project even better.




