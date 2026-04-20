from pathlib import Path

from src.config import PROJECT_ROOT, load_yaml_config
from src.data.loader import load_dataset_split
from src.data.preprocess import PilotBenchmarkConfig, build_triviaqa_pilot_frame


def main() -> None:
    dataset_config = load_yaml_config("dataset_config.yaml")
    dataset_settings = dataset_config.get("dataset", {})

    split_name = dataset_settings.get("pilot_split", "train")
    sample_size = int(dataset_settings.get("pilot_sample_size", 100))
    random_seed = int(dataset_settings.get("random_seed", 42))

    dataset_split = load_dataset_split(
        dataset_name=dataset_settings.get("name", "trivia_qa"),
        config_name=dataset_settings.get("config_name", "rc"),
        split=split_name,
    )

    pilot_config = PilotBenchmarkConfig(
        sample_size=sample_size,
        random_seed=random_seed,
        top_search_results=int(dataset_settings.get("top_search_results", 3)),
        top_entity_pages=int(dataset_settings.get("top_entity_pages", 2)),
    )
    frame = build_triviaqa_pilot_frame(dataset_split, pilot_config)

    output_path = PROJECT_ROOT / "data" / "benchmark" / "triviaqa_pilot_v1.csv"
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"Wrote {len(frame)} pilot rows to {output_path}")


if __name__ == "__main__":
    main()
