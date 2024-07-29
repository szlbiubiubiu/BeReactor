# import whisper
# import torch


# if __name__=='__main__':

#     #audio to text
#     audio_file="audio.mp3"
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_file)
#     print(result["text"])

#     #filter text

#     #LLM processing
# from transformers import pipeline
# import torch
# import pdb
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )

# messages = [
#     {"role": "system", "content": "You are a patient with Depression"},
#     # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?Please reply in Chinese"},
# ]

# terminators = [
#     pipe.tokenizer.eos_token_id,
#     pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipe(
#     messages,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# assistant_response = outputs[0]["generated_text"][-1]["content"]
# print(assistant_response)


# import torch
# from transformers import pipeline
# import pdb

# chat = [
#     {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
#     {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
# ]

# pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
# response = pipe(chat, max_new_tokens=512)
# print(response[0]['generated_text'][-1]['content'])
# pdb.set_trace()

# chat = response[0]['generated_text']
# chat.append(
#     {"role": "user", "content": "Wait, what's so wild about soup cans?"}
# )
# response = pipe(chat, max_new_tokens=512)
# print(response[0]['generated_text'][-1]['content'])

######text text to audio---------------------
# import asyncio

# import edge_tts
# import pdb
# TEXT = "Hello World!"
# VOICE = "en-GB-SoniaNeural"
# OUTPUT_FILE = "test.mp3"
# communicate = edge_tts.Communicate(TEXT, VOICE)
# # with open(OUTPUT_FILE, "wb") as file:
# pdb.set_trace()

# async def amain() -> None:
#     """Main function"""
#     communicate = edge_tts.Communicate(TEXT, VOICE)
#     # with open(OUTPUT_FILE, "wb") as file:
#     pdb.set_trace()
#     async for chunk in communicate.stream():
#         if chunk["type"] == "audio":
#             # file.write(chunk["data"])
#             print(chunk['data'])
#         elif chunk["type"] == "WordBoundary":
#             print(f"WordBoundary: {chunk}")


# if __name__ == "__main__":
#     asyncio.run(amain())

#chattts --------------------------------
# import ChatTTS
# import torch
# import torchaudio
# import pdb
# chat = ChatTTS.Chat()
# chat.load(compile=False) # Set to True for better performance

# texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

# wavs = chat.infer(texts)
# pdb.set_trace()
# for i in range(len(wavs)):
#     torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)

#talking head--------------------------------

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
# from talkinghead.muse.musetalk.utils.utils import get_file_type,get_video_fps,datagen
# from talkinghead.muse.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
# from talkinghead.muse.musetalk.utils.blending import get_image
# from talkinghead.muse.musetalk.utils.utils import load_all_model
import shutil
import torch
import cv2
import pdb
# load model weights
pdb.set_trace()
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

fps=25
wavs='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/1.wav'
img_path='/homes/zs003/test_images/superv.jpg'

whisper_feature = audio_processor.audio2feat(wavs)
whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

#image processing
crop_frame = cv2.imread(img_path)
crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
latents = vae.get_latents_for_unet(crop_frame)
# gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
# for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
whisper_batch=whisper_chunks
audio_feature_batch = torch.from_numpy(whisper_batch)
audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                dtype=unet.model.dtype) # torch, B, 5*N,384
audio_feature_batch = pe(audio_feature_batch)
latent_batch=latents
latent_batch = latent_batch.to(dtype=unet.model.dtype)

pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
recon = vae.decode_latents(pred_latents)
res_frame_list=[]
for res_frame in recon:
    res_frame_list.append(res_frame)
