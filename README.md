# Homekit Pairing Code OCR

This respsitory contains a python based OCR algorithm to extract the homekit pairing code from an image and return it as a string.

## Installation/Dependencies
```
sudo apt-get install python-opencv
pip install numpy
```

## Overview
- **extractor.py:** Takes an image of a homekit device as input, extracts the code box and saves it as a new image to the disk
- **training.py:** Takes an extracted image from extractor.py as input, and starts a user-input based machine learning session
- **production.py:** Takes an extracted image from extractor.py as input, and uses an OCR algorithm to extract the pairing code from it

## Training
Create as many random numbers as you want to and print all of them surrounded with a single box in the Scancardium font. Make a picture of this document. For an example look at training.JPG. Adjust the code_height variable in the extrator.py to return a code_image.jpg file, where the numbers have a height of round about 50px. Run training.py. After the training is completed, change the code_height variable back to 200px.

## Usage
Run extrator.py to extract the pairing code box from the picture. Following use production.py to run the OCR algorithm.

## Example
- **Original image:** production.JPG
- **Extracted code block:** code_block.jpg
- **OCR return:** 601-29-139
- **Training image:** training.JPG
