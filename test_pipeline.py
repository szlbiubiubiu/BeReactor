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
from transformers import pipeline
import torch
import pdb
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a patient with Depression"},
    # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?Please reply in Chinese"},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)