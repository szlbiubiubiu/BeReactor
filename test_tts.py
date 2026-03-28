import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoTokenizer, set_seed

from parler_tts import ParlerTTSForConditionalGeneration

# TODO: Adapt the following as per your requiremens
init_audio_file = "/data/szl/dataset/react2024/test/Audio_files/NoXI/002_2016-03-17_Paris/Expert_video/5.wav"
init_prompt = "Here, write the transcript of the init audio"
description = (
    "A man speaker speaks quickly with a low-pitched voice. "
    "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
)
prompt = "Is it really working ?should I write more description .   I am not sure how much description I should write to get good results ?"

# Load the Models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "parler-tts/parler-tts-mini-v1"
model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

SAMPLING_RATE = model.config.sampling_rate

# Load the init audio
init_audio, init_sr = torchaudio.load(init_audio_file)
init_audio = torchaudio.functional.resample(init_audio, init_sr, SAMPLING_RATE)
init_audio = init_audio.mean(0)  # Take the mean across the channel dim
# Encode the init audio using the feature extractor
input_values = feature_extractor(init_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_values.to(device)


input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
# NOTE: concatenate the init_propt and prompt when passing into the model
prompt_input_ids = tokenizer(init_prompt + " " + prompt, return_tensors="pt").input_ids.to(device)
set_seed(2)

# Generate the audio
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, input_values=input_values)

# Save the audio
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, SAMPLING_RATE)