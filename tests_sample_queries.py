import requests


BASE_URL = "http://127.0.0.1:8000"


QUESTIONS = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]


def main():
    for i, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {i}")
        print(f"Question: {question}")
        print("=" * 70)

        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"query": question},
                timeout=60,
            )

            print(f"HTTP Status: {response.status_code}")

            if response.ok:
                result = response.json()

                print("\nAnswer:")
                print(result.get("final_answer"))

                print("\nConfidence Score:")
                print(result.get("confidence_score"))

                print("\nRetrieved Context:")
                for context in result.get("retrieved_context", []):
                    print(context)
            else:
                print("\nError:")
                print(response.text)

        except requests.RequestException as exc:
            print(f"\nRequest failed: {exc}")


if __name__ == "__main__":
    main()