
# ROBOSRT : A Single Armed Robot for object detection and sorting 

In industrial tasks, the inefficiency and challenges faced by manual labor highlight
the urgent need for innovative solutions. Human vision's limitations and
susceptibility to fatigue further underscore this demand. In today's context, there's a
growing demand for robots equipped with high accuracy, productivity, and error-
free performance, particularly in repetitive tasks like sorting objects in
manufacturing industries.  Our aim is to
develop a system capable of classifying objects based on their properties, such as
color, shape, and texture. This system will analyze video inputs, converting them
into frames for processing and sorting. By addressing these challenges, we seek to
enhance efficiency and productivity in industrial settings while minimizing errors
and labor fatigue.  Overall, our goal is to revolutionize industrial sorting
processes, paving the way for increased efficiency and streamlined operations. 


## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Methodology](#methodology)
4. [Installation](#installation)
5. [Run My Robot](#run-my-robot)
6. [Feedback](#feedback)




## Features

###Operational Modes
-  Manual
-  Semi
-  Auto modes

### Simple User Interface
- Controlled by Mobile Phone

 ##### Used Machine Learning Technique
- Object detection and Sorting 


## Prerequisites

### Hardware Requirements
- **Raspberry Pi 4 B**: A powerful and versatile single-board computer.
- **Adafruit 12bit 16 Channel**: PCB board with Driver for Handling Raspberry pi and Servo Motots
- **Sensors**: Camera Module.
- **Motors**: Servo Motors
- **Robotic Arm**: Full fledged Single Robotic Arm 

### Software Requirements
- **Python 3.8+**: The programming language used for the project.

### Tools
- **VS Code** or **Thonny**
- **Real VNC Viewer** - Development Environment of Raspberry on Personal Laptop or Computer  


## Methodology

System Architecture of Robosort

https://github.com/sriprada346/Robosort/assets/56331169/51918aab-bf22-496d-96d2-2d9d13343da9




## Installation

Unzip the folder and open Robosort folder

Install Robosort Project

```bash
  pip install -r requirements.txt
```
Install **puTTy** for SSH Certificate Enabling





    
## Run My Robot
Follow these steps to run your robot:

1. **Connect to the Same Network**:
    Ensure that both your Raspberry Pi and your mobile device are connected to the same local network.

2. **Start the Web Server**:
    On your Raspberry Pi, open a terminal and run the following command to start a simple HTTP server:
    ```sh
    python -m http.server 8001
    ```

3. **Access the Web Page**:
    On your mobile device, open a web browser and enter the IPv4 address of your Raspberry Pi followed by `:8001`. For example:
    ```
    http://<Raspberry_Pi_IPv4_Address>:8001
    ```
    Replace `<Raspberry_Pi_IPv4_Address>` with the actual IPv4 address of your Raspberry Pi.

4. **Run the Server**:
    On your Raspberry Pi, in another terminal window, run the following command to start the main server application:
    ```sh
    python main.py
    ```

## Feedback

If you have any feedback, please reach out to us at sriprada346@gmail.com



