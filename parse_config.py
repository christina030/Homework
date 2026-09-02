import yaml

def load_config(file_path="config.yaml"):
    try:
        with open(file_path, "r") as file:
            # safe_load prevents arbitrary code execution vulnerabilities
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None
