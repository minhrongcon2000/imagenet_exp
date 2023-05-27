import argparse
import os

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--project_name", type=str, required=True)
parser.add_argument("--max_version", type=int, required=True)
parser.add_argument("--artifact_prefix", type=str, required=True)
args = vars(parser.parse_args())

os.environ["WANDB_API_KEY"] = args["wandb_api_key"]

api = wandb.Api(overrides=dict(project=args["project_name"]))

for i in range(args["max_version"]):
    print(f"Delete version {i}")
    artifact = api.artifact(f"{args['artifact_prefix']}:v{i}", type="model")
    artifact.delete()
