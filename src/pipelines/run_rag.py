from src.config import load_yaml_config


def main() -> None:
    prompt_config = load_yaml_config("prompt_config.yaml")
    print("RAG scaffold is ready.")
    print(f"Prompt configured: {bool(prompt_config.get('rag'))}")


if __name__ == "__main__":
    main()
