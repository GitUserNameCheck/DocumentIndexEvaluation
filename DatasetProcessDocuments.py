import json
import httpx

INPUT_JSONL = "C:/Users/howto/Downloads/SemanticSearch/AnswerEvaluation/LongDocUrl_uploaded_documents.jsonl"   # output from previous script
PROCESS_URL = "http://localhost:5001/api/document/process"
ACCESS_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6InF3ZXJ0eSIsImV4cCI6MTc2OTY1MTE3M30.IxZU_nhVgQGtavZC4Wujlgad9K9RhzZIR5VWlK-TKNQ"

def main():
    client = httpx.Client(
        timeout=500.0,
        cookies={"access_token": ACCESS_TOKEN},
    )

    try:
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_no}: invalid JSON ({e})")
                    continue

                doc_id = record.get("id")
                doc_name = record.get("name")

                print(f"Processing document id={doc_id} ({doc_name})")

                response = client.post(
                    PROCESS_URL,
                    params={"id": doc_id}
                )

                if response.is_success:
                    print(f"Processed id={doc_id}")
                else:
                    # if response.status_code != 409:
                        print(
                            f"Failed to process id={doc_id} "
                            f"({response.status_code}): {response.text}"
                        )

    finally:
        client.close()


if __name__ == "__main__":
    main()