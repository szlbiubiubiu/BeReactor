import torch
from transformers import pipeline
import whisper
import pdb

# import edge_tts
import ChatTTS
# from transformers import pipeline


if __name__=='__main__':

    #text_audio_define
    VOICE = "en-GB-SoniaNeural"
    #load audio
    #audio to text
    audio_file="audio.mp3"
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    print(result["text"])

    #split audio, results --> sentences

    ##LLM processing

    #initlize chat
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    

    # chat = [
    # {"role": "system", "content": "You are a random man with different personallity."},
    # {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    #     ]
    for i in range(len(sentences)):
        if i==0:
            chat = [
            {"role": "system", "content": "You are a random man with different personallity."},
            {"role": "user", "content": sentences[0]}
                ]
        else:
            chat = response[0]['generated_text']
            chat.append(
                {"role": "user", "content": sentences[i+1]}
            )
        response = pipe(chat, max_new_tokens=512)
        current_text=response[0]['generated_text'][-1]['content']
        print(response[0]['generated_text'][-1]['content'])
        
        #text to audio
        
        wavs = chat.infer(current_text)



        # communicate = edge_tts.Communicate(current_text, VOICE)
        # # message in self.stream()
        # for chunk in communicate.stream():
        #     if chunk["type"] == "audio":
        #         file.write(chunk["data"])
        # # communicate.save_sync(OUTPUT_FILE)# change to talking head format

        # #audio to talking head, whisper,tiny

        #talking head
        from musetalk.utils.utils import get_file_type,get_video_fps,datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
        from musetalk.utils.blending import get_image
        from musetalk.utils.utils import load_all_model
        import shutil

        # load model weights
        audio_processor, vae, unet, pe = load_all_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        timesteps = torch.tensor([0], device=device)

        fps=25
        whisper_feature = audio_processor.audio2feat(wavs)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

        #image processing
        crop_frame = cv2.imread(img_path)
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