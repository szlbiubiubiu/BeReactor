import os
import argparse
import platform
import re

try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model
import torch
import torch.nn.functional as F
import whisper
from openai import OpenAI
from omegaconf import OmegaConf

from datasets import get_dataset
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate_LLM(dialogue: str):
    """
    Evaluates the realism of a dialogue using an LLM with chain-of-thought prompting.
    Returns (realism_score, sentiment_emb).
    """
    prompt = (
        "You are a dialogue evaluation expert. Your task is to assess the realism of a given dialogue. "
        "Realism includes factors such as coherence, natural language usage, contextual relevance, and emotional nuance. "
        "Please follow these steps:\n\n"
        "1. Analyze the dialogue text and explain step-by-step your reasoning regarding its realism. "
        "Consider the flow, tone, and whether it sounds like a human conversation.\n"
        "2. Based on your analysis, provide a final realism score on a scale from 1 (not realistic) to 5 (highly realistic).\n\n"
        "3. Based on your analysis, provide a sentiment analysis by choosing number from 'Neutral(0), Happy(1), Sad(2), Surprise(3), Fear(4), Disgust(5), Anger(6), Contempt(7)', and give valence and arousal score(both range from 0-1).\n"
        "Dialogue:\n"
        f"'''{dialogue}'''\n\n"
        "Please output your answer in JSON format as follows:\n"
        "{\n"
        '  "chain_of_thought": "Your step-by-step reasoning here...",\n'
        '  "realism_score": X\n'
        '  "sentiment analysis": X\n'
        '  "valence_score": X\n'
        '  "arousal_score": X\n'
        "}\n"
    )
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini-ca",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500,
        n=1,
        stop=None,
    )

    answer = response.choices[0].message.content.strip()

    arousal_score = re.findall(r'\d+\.\d+', answer.split('\n')[-2])
    valence_score = re.findall(r'\d+\.\d+', answer.split('\n')[-3])
    sentiment_score = re.findall(r'\d+', answer.split('\n')[-4])
    realism_score = re.findall(r'\d+', answer.split('\n')[-5])

    arousal_score = [float(num) for num in arousal_score]
    valence_score = [float(num) for num in valence_score]
    sentiment_score = [int(num) for num in sentiment_score]
    realism_score = [int(num) for num in realism_score]

    if len(arousal_score) == 0:
        arousal_score = 0
        valence_score = 0
        sentiment_score = 0
        realism_score = 0
    elif len(realism_score) == 1:
        arousal_score = arousal_score[0]
        valence_score = valence_score[0]
        sentiment_score = sentiment_score[0]
        realism_score = realism_score[0]

    sentiment_emb = F.one_hot(torch.tensor(sentiment_score), num_classes=10)
    sentiment_emb[-2] = valence_score
    sentiment_emb[-1] = arousal_score
    sentiment_emb = torch.unsqueeze(sentiment_emb, dim=0)

    return realism_score, sentiment_emb


def main(args):
    if 'sbg' in platform.node() or 'dg' in platform.node() or 'xlg' in platform.node():
        args.dataset_path = '/data/home/acw569/paper_face/face_react/latte/latte_reaction/val_test_matrix'
    else:
        args.dataset_path = './val_test_matrix'

    whisper_model = whisper.load_model("base.en")

    dataset = get_dataset(args, train_mode='test')
    batch_size = int(args.local_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    all_realism_scores = []
    all_sentiment_embs = []
    total_samples = 0

    for step, video_data in enumerate(loader):
        print(f'Step {step}, batch_size {batch_size}')
        for batch_index in range(batch_size):
            speaker_audio_path = video_data['speaker_audio_path'][batch_index]
            listener_audio_path = video_data['listener_audio_path'][batch_index]

            speaker_result = whisper_model.transcribe(speaker_audio_path)
            listener_result = whisper_model.transcribe(listener_audio_path)

            speaker_text = speaker_result["text"].strip()
            listener_text = listener_result["text"].strip()

            dialogue = f"Speaker: {speaker_text}\nListener: {listener_text}"
            print(f'[Step {step}][{batch_index}] Speaker: {speaker_text!r}')
            print(f'[Step {step}][{batch_index}] Listener: {listener_text!r}')

            realism_score, sentiment_emb = evaluate_LLM(dialogue)
            print(f'[Step {step}][{batch_index}] realism_score={realism_score}  sentiment_emb={sentiment_emb}')

            all_realism_scores.append(realism_score)
            all_sentiment_embs.append(sentiment_emb)
            total_samples += 1

    if total_samples == 0:
        print("No samples evaluated.")
        return

    avg_realism = sum(all_realism_scores) / total_samples
    avg_sentiment_emb = torch.cat(all_sentiment_embs, dim=0).float().mean(dim=0)

    print("\n========== Language Evaluation Results ==========")
    print(f"Total samples: {total_samples}")
    print(f"Average realism score: {avg_realism:.4f}")
    print(f"Average valence:       {avg_sentiment_emb[-2].item():.4f}")
    print(f"Average arousal:       {avg_sentiment_emb[-1].item():.4f}")
    print(f"Average sentiment emb: {avg_sentiment_emb.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/reaction/reaction_test.yaml")
    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI-compatible API key")
    parser.add_argument("--api_base_url", type=str, default="https://api.chatanywhere.tech/v1",
                        help="Base URL for the OpenAI-compatible API endpoint")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.api_base_url

    omega_conf = OmegaConf.load(args.config)
    main(omega_conf)
