
import pdb
# ####support vllm
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95)

llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",max_model_len=32768)


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



# # #vllm offical example


# # SPDX-License-Identifier: Apache-2.0

# from vllm import LLM, SamplingParams

# # llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",max_model_len=104576)#,max_model_len=32768
# llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",max_model_len=23632)#,max_model_len=32768

# sampling_params = SamplingParams(temperature=0.6,top_p=0.95)


# def print_outputs(outputs):
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#     print("-" * 80)


# print("=" * 80)

# # In this script, we demonstrate how to pass input to the chat method:

# conversation = [
#     # {
#     #     "role": "system",
#     #     "content": "You are a helpful assistant"
#     # },
#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?"
#     },
#     {
#         "role": "user",
#         "content": "Please Write an essay about the importance of higher education.",
#     },
# ]
# outputs = llm.chat(conversation,
#                    sampling_params=sampling_params,
#                    use_tqdm=False)
# print_outputs(outputs)

# pdb.set_trace()
# a=1



# # # You can run batch inference with llm.chat API
# # conversation = [
# #     {
# #         "role": "system",
# #         "content": "You are a helpful assistant"
# #     },
# #     {
# #         "role": "user",
# #         "content": "Hello"
# #     },
# #     {
# #         "role": "assistant",
# #         "content": "Hello! How can I assist you today?"
# #     },
# #     {
# #         "role": "user",
# #         "content": "Write an essay about the importance of higher education.",
# #     },
# # ]
# # conversations = [conversation for _ in range(10)]

# # # We turn on tqdm progress bar to verify it's indeed running batch inference
# # outputs = llm.chat(messages=conversations,
# #                    sampling_params=sampling_params,
# #                    use_tqdm=True)
# # print_outputs(outputs)

# # A chat template can be optionally supplied.
# # If not, the model will use its default chat template.

# # with open('template_falcon_180b.jinja', "r") as f:
# #     chat_template = f.read()

# # outputs = llm.chat(
# #     conversations,
# #     sampling_params=sampling_params,
# #     use_tqdm=False,
# #     chat_template=chat_template,
# # )


# conversation = [

#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?"
#     },
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]
# import ollama
# response = ollama.chat(
#     model="deepseek-r1:7b",
#     messages=conversation
#     # messages=[
#     #     {"role": "user", "content": "Explain Newton's second law of motion"},
#     # ],
# )
# print(response["message"]["content"])


from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process



# server_process, port = launch_server_cmd(
#     "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0"
    
# )

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B  --host 0.0.0.0"
    
)
wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")

import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",#deepseek-ai/DeepSeek-R1-Distill-Llama-8B  meta-llama/Meta-Llama-3.1-8B-Instruct
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print_highlight(f"Response: {response}")

# import openai
# client = openai.Client(
#     base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# # Chat completion
# response = client.chat.completions.create(
#     model="default",
#     messages=[
#         {"role": "system", "content": "You are a helpful AI assistant"},
#         {"role": "user", "content": "List 3 countries and their capitals."},
#     ],
#     temperature=0,
#     max_tokens=64,
# )
# print(response)