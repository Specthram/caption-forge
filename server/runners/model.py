"""Job bodies for loading and unloading the VLM (streamed status)."""

from server.jobs import Progress
from src import loader


def load_body(model_cfg: dict, name: str):
    """Return a job body that loads ``model_cfg``, streaming its status."""

    def run(progress: Progress) -> dict:
        for status, _loaded in loader.load_model(model_cfg):
            progress(sub=status)
        return {"loaded": loader.is_model_loaded(), "name": name}

    return run


def unload_body():
    """Return a job body that unloads the current VLM, streaming status."""

    def run(progress: Progress) -> dict:
        for status, _loaded in loader.unload_model():
            progress(sub=status)
        return {"loaded": loader.is_model_loaded()}

    return run
