"""Job bodies for loading and unloading the VLM (streamed status)."""

from server.jobs import Progress
from src import loader, model_profiles


def load_body(model_cfg: dict, name: str):
    """Return a job body that loads ``model_cfg``, streaming its status."""

    def run(progress: Progress) -> dict:
        for status, _loaded in loader.load_model(model_cfg):
            progress(sub=status)
        return {"loaded": loader.is_model_loaded(), "name": name}

    return run


def load_profile_body(model_cfg: dict, profile: dict):
    """Return a job body swapping VRAM to a profile's weights.

    Unlike :func:`load_body`, an already-resident model is unloaded first
    (the loader refuses to load over one), and the loaded-profile marker is
    kept in sync so the selectors' status dots are truthful.
    """

    def run(progress: Progress) -> dict:
        if loader.is_model_loaded():
            for status, _loaded in loader.unload_model():
                progress(sub=status)
            model_profiles.set_loaded_id(None)
        for status, _loaded in loader.load_model(model_cfg):
            progress(sub=status)
        if loader.is_model_loaded():
            model_profiles.set_loaded_id(profile["id"])
        return {
            "loaded": loader.is_model_loaded(),
            "name": profile["name"],
            "profile_id": profile["id"],
        }

    return run


def unload_body():
    """Return a job body that unloads the current VLM, streaming status."""

    def run(progress: Progress) -> dict:
        for status, _loaded in loader.unload_model():
            progress(sub=status)
        model_profiles.set_loaded_id(None)
        return {"loaded": loader.is_model_loaded()}

    return run
