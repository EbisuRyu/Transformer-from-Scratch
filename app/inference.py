# Handle importing from src and models
import sys
import os
from pathlib import Path
from glob import glob
import streamlit as st
import torch

file_dir_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory path of the current file
models_dir_path = os.path.join(os.path.dirname(file_dir_path), 'model')  # Set the directory path for models
sources_dir_path = os.path.join(os.path.dirname(file_dir_path), 'sources')  # Set the directory path for sources
# Add the parent directory of 'sources' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.machine_translation import MachineTranslationTransformer
from tokenizers import Tokenizer

def get_last_directory(path):
    parent_dir = os.path.dirname(path).split('\\')[-1]
    return os.path.join(parent_dir, os.path.basename(path))

def list_model_and_tokenizer_paths(directory):
    model_paths = []
    tokenizer_paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.pt'):
                model_paths.append(os.path.join(dirpath, filename))
            elif filename.endswith('.json'):
                tokenizer_paths.append(os.path.join(dirpath, filename))
    return model_paths, tokenizer_paths

def load_models(model_pth):
    # Load the model from checkpoint with specified configurations
    model = MachineTranslationTransformer( 
        d_model = d_model,
        num_layers = num_layers,
        src_vocab_size = vocab_size,
        trg_vocab_size = vocab_size,
        n_heads = n_heads,
        d_ff = d_ff,
        max_len = max_len,
        dropout = 0)
        
    # Load model weights based on the selected device
    if device == 'cuda':
        state = torch.load(model_pth, map_location = torch.device('cuda'))
        model.load_state_dict(state['model_state_dict'])
    else:
        state = torch.load(model_pth, map_location = torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

# Initialize dictionaries to hold model and tokenizer paths
model_dict = {}
tokenizer_dict = {}

# Get lists of model and tokenizer paths
model_paths, tokenizer_paths = list_model_and_tokenizer_paths(models_dir_path)

# For each model path, add an entry to the model dictionary
# The key is the last directory in the path, and the value is the path itself
for model_path in model_paths:
    model_dict[get_last_directory(model_path)] = model_path

# Do the same for the tokenizer paths
for tokenizer_path in tokenizer_paths:
    tokenizer_dict[get_last_directory(tokenizer_path)] = tokenizer_path

# Get lists of the keys (i.e., the last directories in the paths) in the model and tokenizer dictionaries
model_lst = list(model_dict.keys())
tokenizer_lst = list(tokenizer_dict.keys())

#---------------------------------Streamlit App---------------------------------#
# Configure streamlit layout
st.title("Transformer Inference App")  # Set the title of the Streamlit app

# Set up the the sidebar for model and tokenizer selection
st.sidebar.subheader("Model and Tokenizer Selection")  # Add a subheader to the sidebar
selected_model = st.sidebar.selectbox("Model", model_lst)  # Create a dropdown for selecting the model
selected_tokenizer = st.sidebar.selectbox("Tokenizer", tokenizer_lst)  # Create a dropdown for selecting the tokenizer

st.sidebar.subheader("Model Config")  # Add a subheader to the sidebar for model configuration
d_model = st.sidebar.number_input("d_model", step = int(), value = 512)  # Add a numeric input field for setting d_model
num_layers = st.sidebar.number_input("num_layers", step = int(), value = 6)  # Add a numeric input field for setting num_layers
n_heads = st.sidebar.number_input("n_heads", step = int(), value = 8)  # Add a numeric input field for setting n_heads
vocab_size = st.sidebar.number_input("vocab_size", step = int(), value = 60000)  # Add a numeric input field for setting vocab_size
d_ff = st.sidebar.number_input("d_ff", step = int(), value = 2048)  # Add a numeric input field for setting d_ff
max_len = st.sidebar.number_input("max_len", step = int(), value = 500)  # Add a numeric input field for setting max_len
dropout = st.sidebar.number_input("dropout", step = int(), value = 0)  # Add a numeric input field for setting dropout
device = st.sidebar.radio("Device", ("CPU", "GPU"))  # Add a radio button for selecting the device (CPU or GPU)
# Set up the main content for the app
text_input = st.text_input("Input text", "")  # Add a text input field for user input
run_button = st.button("Run model")  # Add a button for running the model


if run_button:
    # Load model and tokenizer
    model_pth = model_dict[selected_model]  # Get the path of the selected model
    tokenizer_pth = tokenizer_dict[selected_tokenizer]  # Get the path of the selected tokenizer
    model = load_models(model_pth)  # Load the selected model
    tokenizer = Tokenizer.from_file(tokenizer_pth)  # Load the selected tokenizer
    out = model.translate(
        text_input,  # Input text to be translated
        tokenizer,  # Tokenizer for tokenizing input text
    )
    st.write(out)  # Display the translation output