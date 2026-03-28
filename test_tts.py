import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import scipy.io.wavfile as wavfile

# 1. 加载模型和 Tokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 定义声音的描述（保持不变）
# 描述越详细，声音越固定
voice_description = "A female speaker with a slightly high-pitched voice delivers her words quite expressively, in a very clear, close-sounding environment."
input_ids = tokenizer(voice_description, return_tensors="pt").input_ids.to(device)

# 3. 定义两句不同的话
text_1 = "Hello! I am speaking with this specific voice."
text_2 = "Now, I am saying something completely different, but my voice should sound the same."

prompt_input_ids_1 = tokenizer(text_1, return_tensors="pt").input_ids.to(device)
prompt_input_ids_2 = tokenizer(text_2, return_tensors="pt").input_ids.to(device)

# 4. 固定随机种子并生成第一段音频
seed = 42 # 选择一个你喜欢的种子

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

generation_1 = model.generate(
    input_ids=input_ids, 
    prompt_input_ids=prompt_input_ids_1
)

# 5. 再次固定相同的随机种子，生成第二段音频
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

generation_2 = model.generate(
    input_ids=input_ids, 
    prompt_input_ids=prompt_input_ids_2
)

# 保存音频文件查看效果
audio_arr_1 = generation_1.cpu().numpy().squeeze()
audio_arr_2 = generation_2.cpu().numpy().squeeze()
sample_rate = model.config.sampling_rate

wavfile.write("output_sentence_1.wav", sample_rate, audio_arr_1)
wavfile.write("output_sentence_2.wav", sample_rate, audio_arr_2)

print("音频已生成！两者应该具有相同的音色。")