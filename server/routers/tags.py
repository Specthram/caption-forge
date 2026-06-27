"""Tag routes: categories (CRUD + reorder), paged catalogue, tag CRUD."""

from fastapi import APIRouter, HTTPException

from server.schemas import (
    CategoryBody,
    CategoryUpdateBody,
    MoveTagBody,
    RenameBody,
    ReorderBody,
    TagCreateBody,
    TagNameBody,
    TagNamesBody,
)
from src import sqlite_store as store

router = APIRouter(prefix="/api/tags", tags=["tags"])


@router.post("/dedupe")
def dedupe_tags() -> dict:
    """Merge tags duplicated across categories into their real one.

    Cleanup for tags the WD14 auto-tagger cloned into its fallback category
    before the reuse fix: the copy in the auto-tag category is merged into the
    tag's real category (its media re-pointed), leaving one row per name.
    """
    reserved = store.uncategorized_category_id()
    merged = store.dedupe_tags(reserved_category_id=reserved or None)
    return {
        "merged": merged,
        "names": len(merged),
        "removed": sum(item["merged"] for item in merged),
    }


@router.post("/existing")
def existing_names(body: TagNamesBody) -> dict:
    """Return which of the given tag names already exist (any category)."""
    return {"existing": store.existing_tag_names(body.names)}


@router.get("/categories")
def list_categories() -> dict:
    """Return tag categories in display order, with per-category counts."""
    return {
        "categories": [
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "count": store.count_tags(category_id=row["id"]),
            }
            for row in store.list_tag_categories()
        ]
    }


@router.post("/categories")
def create_category(body: CategoryBody) -> dict:
    """Create a tag category."""
    try:
        category_id = store.create_tag_category(body.name, body.color)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": category_id}


@router.post("/categories/reorder")
def reorder_categories(body: ReorderBody) -> dict:
    """Persist a new category display order (drives the tags output)."""
    store.reorder_tag_categories(body.ordered_ids)
    return {"ok": True}


@router.post("/categories/{category_id}")
def update_category(category_id: int, body: CategoryUpdateBody) -> dict:
    """Rename and/or recolour a tag category."""
    store.update_tag_category(category_id, name=body.name, color=body.color)
    return {"ok": True}


@router.delete("/categories/{category_id}")
def delete_category(category_id: int) -> dict:
    """Delete a category (cascades to its tags)."""
    store.delete_tag_category(category_id)
    return {"ok": True}


@router.get("/list")
def list_tags(
    category_id: int,
    query: str = "",
    limit: int = 96,
    offset: int = 0,
) -> dict:
    """Return one page of a category's tags with usage counts.

    Paged/virtualised for categories that can hold 150k tags; never returns
    the whole set. Searching ranks by match quality (server-side).
    """
    total = store.count_tags(category_id=category_id, query=query)
    page = store.list_tags_page(category_id, query, limit, offset)
    counts = store.tag_usage_counts([row["id"] for row in page])
    return {
        "total": total,
        "items": [
            {
                "id": row["id"],
                "name": row["name"],
                "usage_count": counts.get(row["id"], 0),
            }
            for row in page
        ],
    }


@router.post("")
def create_tag(body: TagCreateBody) -> dict:
    """Create (or reuse) a tag in a category."""
    try:
        tag_id = store.get_or_create_tag(body.name, body.category_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": tag_id}


@router.post("/uncategorized")
def create_uncategorized_tag(body: TagNameBody) -> dict:
    """Reuse a tag by name, else create it in the "Uncategorized" pen.

    Backs typing a brand-new tag in the Libraries "Bulk tags" box: an
    existing name (in any category) is reused, a new one lands in the
    auto-managed "Uncategorized" category.
    """
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tag name is empty.")
    return {"id": store.get_or_create_tag_reuse(name), "name": name}


@router.post("/{tag_id}/move")
def move_tag(tag_id: int, body: MoveTagBody) -> dict:
    """Move a tag to another category (merging on a name collision)."""
    try:
        kept = store.move_tag(tag_id, body.category_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": kept}


@router.post("/{tag_id}/rename")
def rename_tag(tag_id: int, body: RenameBody) -> dict:
    """Rename a tag."""
    try:
        store.rename_tag(tag_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.delete("/{tag_id}")
def delete_tag(tag_id: int) -> dict:
    """Delete a tag (cascades to its media links)."""
    store.delete_tag(tag_id)
    return {"ok": True}


@router.get("/search")
def search_tags(query: str = "", category_id: int | None = None) -> dict:
    """Return tags matching a live-search query (usage-ranked, capped)."""
    rows = store.search_tags(query, category_id=category_id, limit=50)
    return {
        "tags": [
            {
                "id": row["id"],
                "name": row["name"],
                "category_id": row["category_id"],
                "usage_count": row["usage_count"],
            }
            for row in rows
        ]
    }
