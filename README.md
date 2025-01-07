# Senior-Project
(high school senior project)

A fine-tuned LSTM for translating Middle English to Present-Day English. Based on an open-source German-English LSTM, with custom scraped and translated Middle English data.

---

## **Table of Contents**
1. [Data](#data)  
2. [Model](#model)  
3. [Usage](#usage)  
   - [Installation](#installation)  
4. [Directory Structure](#directory-structure)  

---

## **Dataset**

- ~1000 sentences, with about 50% manually translated by me and a volunteer. The remainder was translated by the authors of *The Riverside Chaucer*.
- To see examples and additional notes, see `dataset outline and notes.pdf` and `Senior Project Presentation of Benjamin Lambright.pdf`

---

## **Models**

### **German to English**:
- The first translator I made by watching a tutorial on how to make a neural machine translator by Aladdin Persson, most of my code for Middle English to Present Day English was adapted from this. Here is the tutorial video: https://www.youtube.com/watch?v=EoGUlvhRYpk.

### **Middle English to Present Day English**:
- The primary code that runs my neural machine translator. Most of this is adapted from open source code I got from the tutorial video. I fine-tuned it and and of course changed it with my custom data.
- The BLUE score of the training data was 64% and test data was about 4%. This was really because I really didn't have much of any data, but it was cool to see how the entire process from data collection/annotation to training worked in high school.

---

## **Usage**

### **Installation**
1. Clone the repository
2. Run `ME-PDE3.py` for Middle English to Present-Day English translation
3. Run `DE-PDE.py` for German to English translation

---

## **Directory Structure**

<pre>
project/
│
├── data/
│   ├── Chaucer/                       # My dataset of interlinear translations from _The Canterbury Tales_.
│   ├── ChaucerMaker.py/               # Code that splits Chaucer to put into my main datasets. 
│   ├── dataset outline and notes.pdf/ # This is where Rachel and I put all of our initial translations, noting where we got them and any other editors. The comments and notes for this can be seen on the google doc in my Senior Project shared folder, as noted in NOTE.
│   ├── test.xlsx/                     # The spreadsheet of the glossed translations of test data.
│   ├── test2016.me/                   # My ME test data.
│   ├── test2016.pde/                  # My PDE test data.
│   ├── train.me/                      # My ME train data.
│   ├── train.pde/                     # My PDE train data.
│   ├── train.xlsx/                    # The spreadsheet of the glossed translations of train data.
│   ├── val.me/                        # My ME validation data.
│   ├── val.pde/                       # My PDE validation data.
│   ├── validation.xlsx/               # The spreadsheet of the glossed translations of validation data.
│  
├── model/
│   ├── DE-PDE.py                     # German to English model
│   ├── ME-PDE3.py                    # German to English model
│   ├── my_checkpoint4.pth.tar        # Checkpoint of trained model
│   ├── utils.py/                     # Utils for the German translator.
│   ├── utils3.py/                    # Utils for the Middle English translator
│
├── README.md                         # Project documentation
│
└── Senior Project Presentation of Benjamin Lambright.pdf
</pre>
