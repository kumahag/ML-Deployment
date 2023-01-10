# ML-Deployment Corporation Favorita Retail Sales to App

Introduction

This project deploys a regression ML on Corporation Favorita Sales into a Web App using streamlit.This project's aim is to learn how to embeded an ML model into a web app with a user-friendly interface, in this case, Streamlit to have an interface that makes it easier for users to interact with an ML model, regardless of prior knowledge in machine learning.

Process Description

The process begins with exporting the necessary items from the notebook, building an interface that works correctly, importing the necessary items for modelling, and then writing the code to process inputs. The process can therefore be summarized as:

Export machine learning items from notebook,
Import the machine learning items into the app script,
Build the interface,
Write backend code to process inputs,
Pass values through the interface,
Recover these values in backend,
Apply the necessary processing,
Submit the processed values to the ML model to make the predictions,
Process the predictions obtained and display them on the interface.

Setup

To setup this project, you need to have Python3 on your system. Then you can clone this repo and being at the repo's root :: repo_name> ... follow the steps below:

Windows:

  python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
Linux & MacOs:

  python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
NB: For MacOs users, please install Xcode if you have an issue.

Execution
Run the Streamlit app (being at the repository root):

  streamlit run src/streamlit_app.py
Go to your browser at the following address :

http://localhost:8501
