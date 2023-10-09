# Unravelling-Temporal-patterns   

The program has the capability to real time categorize time series data into various classes such as periodic, chaotic, noise, hyperchaotic, etc. This classification is performed in real-time using an Arduino Microcontroller. Additionally, the program offers a visualization feature similar to an oscilloscope. It can take input data, such as temperature readings from any system, and directly forecast its behavior.

If the trained model also can distnguish period doubling etc then it should also be able to predict those as well. Models can be trained using https://github.com/am0032/Unravelling-Temporal-Patterns

To utilize the program, you simply need to convert input voltages to a range of 0-5 volts and input them into the A0 pin of an Arduino, which can then be connected to a laptop. The user interface will prompt the user to select a pre-trained model, and subsequently, the predicted class will be displayed in the user interface.

![image](https://github.com/am0032/Real-Time-ML-Classification-of-Time-Series/assets/123314532/316bbc24-763c-43d8-8eb1-3ebc171479d4)

   
 


## Table of Contents
- [Installation](#Installation)
  - [Using-Python](#Using-Python)
- [Usage](#Usage)
- [Issues](#issues)
- [License](#licensing)



## Installation  
Instructions on how to install the project.  



## Using-Python    
1) Download the Python main file and Arduino file.   

2) Install Python from - https://www.python.org/downloads/   
    Tick the button saying add Python to path in the installation window for Python installation.  

    If Python is already installed and was not added to path simply uninstall and perform the above steps.   



3) Open command prompt on Windows:   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/3d5f24b6-00f9-4425-807f-263ece9e9f1a)   

Please copy the commands below (using top right icon below), paste (right click ) them into Windows  command prompt window, and run (Enter) them to install the necessary dependencies :  

```bash
pip install pyserial matplotlib scikit-learn numpy scipy pyrqa joblib kneed pandas python-igraph

```
4) Install Arduino IDE from : https://www.arduino.cc/en/software   
5) Open Arduino software and load the Arduino file downloaded and then select the COM port in tools and hit upload.    
6) Once the libraries are installed double click on the downloaded python file  to run the program.    


## Usage   
Connect Arduino to laptop and connect A0 pin of Arduino to the source signal  
![image](https://github.com/am0032/Real-Time-ML-Classification-of-Time-Series/assets/123314532/a0684b34-be8b-4fbf-afa9-b788254284da)  














