import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from pixl.conversation import default_conversation
from pixl.utils import disable_torch_init


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


@torch.inference_mode()
def eval_model(model_name, questions_file, answers_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()


    ques_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(ques_file)):
        idx = json.loads(line)["question_id"]
        qs = json.loads(line)["text"]
        cat = json.loads(line)["category"]
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        # stopping_criteria = KeywordsStoppingCriteria([conv.sep], tokenizer, input_ids)
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            use_cache=True,
            temperature=0.7,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.eos_token_id,  # Pad token
            # stopping_criteria=[stopping_criteria]
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval_model(args.model_name, args.question_file, args.answers_file)
