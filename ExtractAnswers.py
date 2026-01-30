import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_JSONL = "C:/Users/howto/Downloads/SemanticSearch/AnswerEvaluation/Answers.jsonl"
OUTPUT_JSONL = "C:/Users/howto/Downloads/SemanticSearch/AnswerEvaluation/Extracted_answers.jsonl"
QWEN_PATH = "C:/Users/howto/Downloads/SemanticSearch/MyNewProject/QwenQwen2.5-7B-Instruct"
ANSWER_EXTRACTION_PROMPT = "C:/Users/howto/Downloads/SemanticSearch/AnswerEvaluation/AnswerExtractionPrompt.md"

SYSTEM_PROMPT = (
    "You are an expert in document question answering. Answer the question strictly based on the given document. \n"
)


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():

    qwen_model= AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype="auto",
        device_map="auto"
    )
    print(qwen_model.hf_device_map)
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)

    with open(ANSWER_EXTRACTION_PROMPT) as f:
        extraction_prompt = f.read()

    rows = load_jsonl(INPUT_JSONL)

    existing_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        for row in load_jsonl(OUTPUT_JSONL):
            existing_ids.add(row.get("question_id"))

    for row in rows:
        question_id = row.get("question_id")
        if question_id in existing_ids:
            print(f"Skipping existing question {question_id}")
            continue

        question = row.get("question")
        predicted_answer = row.get("predicted_answer")

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": SYSTEM_PROMPT + extraction_prompt + "\nQuestion: " + question + "\nAnalysis: " + predicted_answer}
        ]

        print("Extracting answer for question " + question_id)

        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)

        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        extractor_result = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            import re
            concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", extractor_result, re.DOTALL)[0]
            answer_format = re.findall(r"<answer_format>(.*?)</answer_format>", extractor_result, re.DOTALL)[0]
        except:
            concise_answer = "Fail to extract"
            answer_format = "None"

        row.setdefault("predicted_concise_answer", concise_answer)
        row.setdefault("predicted_concise_answer_format", answer_format)

        append_jsonl(OUTPUT_JSONL, row)


if __name__ == "__main__":
    main()


