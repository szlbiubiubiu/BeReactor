import torch
from transformers import pipeline
import whisper
import pdb

import edge_tts
# import ChatTTS
import cv2
# from transformers import pipeline

import sys
sys.path.append('/import/nobackup_mmv_ioannisp/zs003/code/face_reaction/latte_reaction/talkinghead/muse')
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
# from talkinghead.muse.musetalk.utils.utils import get_file_type,get_video_fps,datagen
# from talkinghead.muse.musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
# from talkinghead.muse.musetalk.utils.blending import get_image
# from talkinghead.muse.musetalk.utils.utils import load_all_model
import shutil
# import torch
import cv2
import pdb
import numpy as np
from tqdm import tqdm
import urllib.request

if __name__=='__main__':

    #text_audio_define
    VOICE = "en-GB-SoniaNeural"
    audio_file='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/3.wav'
    #load audio
    #audio to text
    # audio_file="audio.mp3"
    model = whisper.load_model("base.en")
    # pdb.set_trace()
    result = model.transcribe(audio_file)
    print(result["text"])


    # audio = whisper.load_audio(audio_file)
    # audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, options)

    # # print the recognized text
    # print(result.text)
    pdb.set_trace()
    #split audio, results --> sentences
    sentences=result["text"].split('.')
    ##LLM processing
    # pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="auto")

    #initlize chat
    
    #init talking head
    
    # load model weights
    # pdb.set_trace()
    audio_processor, vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)

    fps=25
    batch_size=8
    # wavs='/import/nobackup_mmv_ioannisp/zs003/face_react/new_audio_folder/all_together/NoXI/026_2016-04-06_Nottingham/Expert_video/1.wav'
    img_path='/homes/zs003/test_images/superv.jpg'
    # chat = ChatTTS.Chat()
    # chat.load(compile=False) # Set to True for better performance
    # chat = [
    # {"role": "system", "content": "You are a random man with different personallity."},
    # {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    #     ]
    rand_num=torch.randint(0,100,(1,))
    for i in range(len(sentences)):
        if i==0:
            chat = [
            {"role": "system", "content": "You are a person with a specific personality, subjected to the random number:."+str(6)},
            {"role": "user", "content": sentences[0]}
                ]
            a=1
        else:
            chat = response[0]['generated_text']
            chat.append(
                {"role": "user", "content": sentences[i+1]}
            )
        response = pipe(chat, max_new_tokens=512)#,pad_token_id=pipe.tokenizer.eos_token_id
        current_text=response[0]['generated_text'][-1]['content']
        print(response[0]['generated_text'][-1]['content'])
        
        #text to audio
        
        # wavs = chat.infer(current_text)


        pdb.set_trace()
        communicate = edge_tts.Communicate(current_text, VOICE)
        # # message in self.stream()
        wavs=[]
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                # file.write(chunk["data"])
                wavs.append(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                print(f"WordBoundary: {chunk}")
        # communicate.save_sync(OUTPUT_FILE)# change to talking head format

        # #audio to talking head, whisper,tiny

        #talking head
        
        pdb.set_trace()
        whisper_feature = audio_processor.audio2feat(wavs)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

        #image processing
        crop_frame = cv2.imread(img_path)
        crop_frame=cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list=[]
        input_latent_list.append(latents)
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(len(whisper_chunks)/batch_size)))):
        # for whisper_batch in whisper_chunks:
            # whisper_batch=whisper_chunks
            # pdb.set_trace()
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            # latent_batch=latents
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            # pdb.set_trace()
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            res_frame_list=[]
            for res_frame in recon:
                res_frame_list.append(res_frame)
