"""Settings routes: read the effective settings, persist a save."""

from fastapi import APIRouter

from server.schemas import SettingsBody
from src import caption_score, quality
from src import settings as cf_settings
from src import siglip_grounding

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings() -> dict:
    """Return the effective settings plus the standalone UI preferences.

    :func:`src.settings.load_settings` returns the factory-default keys
    overlaid by the user layer; the standalone keys (written outside a
    Settings save) are added explicitly so the front-end has them all;
    ``grounding_sizes`` ships the SigLIP catalogue the Settings tab renders
    its size / resolution selects from, and ``grounding_model_id`` the
    checkpoint they currently resolve to.
    """
    data = cf_settings.load_settings()
    data.update(
        {
            "quality_display_metric": (
                cf_settings.get_quality_display_metric()
            ),
            "last_caption_type": cf_settings.get_last_caption_type(),
            "review_after_generation": (
                cf_settings.get_review_after_generation()
            ),
            "grounding_model_id": cf_settings.get_grounding_model_id(),
            "grounding_sizes": siglip_grounding.MODEL_SIZES,
            "caption_score_catalogue": caption_score.CATALOGUE,
            "caption_score_clip_id": (cf_settings.get_caption_score_clip_id()),
            "caption_score_blip_id": (cf_settings.get_caption_score_blip_id()),
            "quality_metrics_catalogue": [
                {
                    "id": metric_id,
                    "label": metric.label,
                    "vram": metric.vram_note,
                }
                for metric_id, metric in quality.QUALITY_METRICS.items()
            ],
        }
    )
    return data


@router.post("")
def save_settings(body: SettingsBody) -> dict:
    """Persist a settings save (diffed against the defaults by src)."""
    cf_settings.save_settings(body.settings)
    # Re-export the HF token so a change takes effect without a restart.
    cf_settings.apply_hf_token()
    return {"ok": True}
