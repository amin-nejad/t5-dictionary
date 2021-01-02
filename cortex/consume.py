# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cortex
import requests


# %%
api_spec = {
    "name": "t5",
    "kind": "RealtimeAPI",
    "predictor": {"type": "python", "path": "predictor.py"},
}

cx = cortex.client("gcp")
cx.create_api(api_spec, project_dir=".")


# %%
endpoint = cx.get_api("text-generator2")["endpoint"]
print(endpoint)


# %%
payload = {"text": "translate english to german: who even are you?"}
print(requests.post(endpoint, payload).text)
