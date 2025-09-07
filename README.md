# **Medical Prescription Text Detection Application(final version)**

This application is used for detecting printed and handwritten text regions of a medical prescription image.
This model supports texts in English and Farsi(Persian) language.

CHECK THE EXPERIMENTAL VERSION FOR THE WHOLE PROCESS --> https://github.com/Raminydo/Prescription-Text-Detection


ATTENTION: please turn on your internet connection during usage!






### Application Overview

![base](./screenshots/base.png)

![result](./screenshots/result.png)

## **Preparation**

## prerequisites
- Python 3.11.9
- pip 25.1.1

## Installation
1. **Create virtual Environment**

```bash
  python -m venv .venv
  .\.venv\Scripts\activate    # Windows
  source .venv/bin/activate   # Linux/Mac
```

2. **Clone and Setup**

3. **Install Dependencies**
```bash
  pip install -r requirements.txt
```

## **Usage Guide**
Run application with GUI(web app):
```bash
  streamlit run UI\main.py
```


    

## **Inside The Model**

The approach used in this application has two main steps:

- Detecting all text areas
- Classify the detected areas into printed or handwritten

This application contains 1 method for detecting the text regions:
    
- Detector 1: PaddleOCR (default)

all tested methods in experimental version:
- contours and EasyOCR
- contours, adaptivethresholding and morphology (version 1)
- contours, adaptivethresholding and morphology (version 2)
- merge of two detectors (version 1)
- merge of two detectors (version 2)
- Selective Search
- Faster r-cnn
- CRAFT
- trOCR
- docTR


## **output sample**
![App Screenshot](./screenshots/output.png)

## **Folder Structure**
This structure is used when the project was getting worked on.
```text
project/
|
|--- CLASSIFICATION/
|     |--- processing.py
|
|--- PREPROCESSING/
|     |--- detection.py
|     |--- feature_extraction.py
|
|--- screenshots/
|
|--- TEMP/
|
|--- UI/
|     |--- main.py
|
|--- boosted_model.pkl
|--- README.md
|--- requirements.txt
```
