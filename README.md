# PlanetIdleMinerYOLOV8OD



**PlanetIdleMinerYOLOV8OD** is a project that combines **YOLOv8 object detection** with **Python automation** to autonomously play a mobile Planet Miner Idle game.  
The system detects in-game objects (buttons, resources, upgrades) in real time and interacts with them automatically, showcasing skills in computer vision, machine learning, and Python-based automation.

Although I am familiar with opencv template matching or contouring, (which can be more efficient), I wanted to play with object detection models. I annotated all images myself, processed them and trained the models using the T4+ GPU from google collab.

## Features
- **YOLOv8 Object Detection**  
  Trained models detect key UI elements and objects in the game.
- **Automated Gameplay**  
  Python scripts interact with detected objects (e.g., tap, collect, upgrade).
- **Real-Time Integration**  
  Processes live gameplay frames for instant interaction.
