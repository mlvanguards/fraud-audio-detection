import streamlit as st
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import tempfile
from models import CRNNWithAttn

# Create the model and put it on the GPU if available
myModel = CRNNWithAttn()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

# Load your model
@st.cache_resource
def load_model():
    myModel.load_state_dict(torch.load('./models/best_model10.pth', map_location=device))
    myModel.eval()
    return myModel

model = load_model()

# ImageNet normalization stats (for 1 channel input)
imagenet_mean = torch.tensor([0.485]).view(1, 1, 1, 1)
imagenet_std = torch.tensor([0.229]).view(1, 1, 1, 1)

# Audio preprocessing
def preprocess(waveform, sample_rate):
    # Resample if needed
    if sample_rate != 16000:
        resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    # Mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Trim / pad to 4 seconds
    max_len = 16000 * 4
    all_waveforms = []
    for _ in range(waveform.shape[1] // max_len + 1):
      if waveform.shape[1] > max_len:
          all_waveforms.append(waveform[:, :max_len])
          waveform = waveform[:, max_len:]
      elif waveform.shape[1] < max_len:
          pad_len = max_len - waveform.shape[1]
          all_waveforms.append(torch.nn.functional.pad(waveform, (0, pad_len)))

    # Convert to MelSpectrogram
    all_spec = []
    for waveform in all_waveforms:
      mel_spec = T.MelSpectrogram(
          sample_rate=16000,
          n_fft=780,
          hop_length=195,
          n_mels=64
      )(waveform)
      mel_spec = T.AmplitudeToDB(top_db=80)(mel_spec)

      all_spec.append(mel_spec)
    return all_spec

# Inject custom CSS styles
st.markdown("""
    <style>
        /* Global background and font */
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }

        .reportview-container {
            background-color: #f9f9f9;
        }

        /* Center elements */
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        /* Styled prediction card */
        .prediction-box {
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            background-color: #ffffff;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            width: 80%;
            text-align: center;
        }

        .prediction-label {
            font-size: 26px;
            font-weight: bold;
        }

        .confidence {
            font-size: 18px;
            color: #555;
            margin-top: 5px;
        }

        .label-real {
            color: green;
        }

        .label-fake {
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='centered'>", unsafe_allow_html=True)
st.title("🔊 Audio Classifier: REAL or FAKE")
st.markdown("<p style='margin-bottom:20px;'>Upload a audio file and let the model classify it!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a .wav/.flac file", type=["wav", "flac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    predicted_classes = []
    predicted = []
    waveform, sample_rate = torchaudio.load(tmp_file_path, backend="soundfile")
    input_tensors = preprocess(waveform, sample_rate)
    for input_tensor in input_tensors:
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor = (input_tensor - imagenet_mean) / imagenet_std
        with torch.no_grad():
            file_index = 0  # to track file names
            outputs = model(input_tensor)
            predicted_classes.append(torch.sigmoid(outputs))
            predicted.append((torch.sigmoid(outputs) > 0.5).int())

    label = "Real" if torch.mean(torch.cat(predicted_classes)) >= 0.4 else "Fake"
    # Display
    st.audio(uploaded_file, format="audio/wav")
    label_class = "label-real" if label == "Real" else "label-fake"
    st.markdown(f"""
        <div class='prediction-box'>
            <div class='prediction-label {label_class}'>Prediction: {label}</div>
            <div class='confidence'>Confidence Score: <code>{torch.mean(torch.cat(predicted_classes)):.2f}</code></div>
        </div>
    """, unsafe_allow_html=True)
