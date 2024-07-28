import torch
from transformers import pipeline
import whisper
import pdb

import edge_tts

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
        communicate = edge_tts.Communicate(current_text, VOICE)
        # message in self.stream()
        for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
        # communicate.save_sync(OUTPUT_FILE)# change to talking head format

        #audio to talking head, whisper,tiny