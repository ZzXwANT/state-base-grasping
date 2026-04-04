import wandb
import requests

api = wandb.Api()
runs = api.runs("zixinzhao-shandong-university/state_full_obs")

api_key = api.api_key

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

mutation = """
mutation UpdateRun($entity: String!, $project: String!, $name: String!) {
  upsertBucket(input: {entityName: $entity, modelName: $project, name: $name, state: "finished"}) {
    bucket { state }
  }
}
"""

for run in runs:
    if run.state in ("crashed", "running"):
        print(f"Fixing: {run.name} ({run.state})")
        resp = requests.post(
            "https://api.wandb.ai/graphql",
            json={
                "query": mutation,
                "variables": {
                    "entity": "zixinzhao-shandong-university",
                    "project": "state_full_obs",
                    "name": run.name,
                },
            },
            headers=headers,
        )
        data = resp.json()
        if "errors" in data:
            print(f"  -> ERROR: {data['errors']}")
        else:
            state = data["data"]["upsertBucket"]["bucket"]["state"]
            print(f"  -> state: {state}")