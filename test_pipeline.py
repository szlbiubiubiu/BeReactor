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
import ChatTTS
import torch
import torchaudio
import pdb
chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
