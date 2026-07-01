import json

from model_registry import registry_rows


if __name__ == "__main__":
    print(json.dumps(registry_rows(), indent=2))
