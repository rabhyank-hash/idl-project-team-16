import json
import os
from pathlib import Path
from datasets import Dataset
def default_image_root()->str:
    local_root=os.environ.get("LOCAL")
    if local_root:
        return str(Path(local_root) / "project_data")
    return "./data/images"
def default_jsonl_path()->str:
    return "./vanilla_matched_6445.jsonl"
def resolve_data_paths(jsonl_path: str | None=None,image_root: str | None=None)->tuple[str,str]:
    default_jsonl=default_jsonl_path()
    default_image=default_image_root()
    resolved_jsonl=jsonl_path or default_jsonl
    resolved_image_root=image_root or default_image
    if not Path(resolved_jsonl).exists() and Path(default_jsonl).exists():
        resolved_jsonl=default_jsonl
    if not Path(resolved_image_root).exists() and Path(default_image).exists():
        resolved_image_root=default_image
    if not Path(resolved_jsonl).exists() and Path("./data/vanilla_matched_6445.jsonl").exists():
        resolved_jsonl="./data/vanilla_matched_6445.jsonl"
    if not Path(resolved_image_root).exists() and Path("./data/images").exists():
        resolved_image_root="./data/images"
    return resolved_jsonl,resolved_image_root
def _resolve_image_for_record(record: dict,image_root: Path)->str | None:
    dataset_index=record.get("dataset_index")
    if dataset_index is not None:
        for ext in (".jpg",".png"):
            candidate=image_root / f"{dataset_index}{ext}"
            if candidate.exists():
                return str(candidate.resolve())
    if not hasattr(_resolve_image_for_record,"_manifest_cache"):
        _resolve_image_for_record._manifest_cache={}
    cache_key=str(image_root)
    if cache_key not in _resolve_image_for_record._manifest_cache:
        manifests=sorted(image_root.rglob("manifest.jsonl"))
        if manifests:
            lookup={}
            manifest_path=manifests[0]
            with manifest_path.open("r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    entry=json.loads(line)
                    filename=entry.get("filename") or entry.get("original_path")
                    if not filename:
                        continue
                    img_path=manifest_path.parent / filename
                    for key in (
                        entry.get("dataset_index"),
                        entry.get("annotation_id"),
                        entry.get("source_annotation_id"),
                    ):
                        if key is not None:
                            lookup[str(key)]=str(img_path)
            _resolve_image_for_record._manifest_cache[cache_key]=lookup
        else:
            _resolve_image_for_record._manifest_cache[cache_key]={}
    manifest_lookup=_resolve_image_for_record._manifest_cache[cache_key]
    if manifest_lookup:
        record_key=str(
            record.get("dataset_index")
            or record.get("annotation_id")
            or record.get("source_annotation_id")
            or ""
        )
        img_path=manifest_lookup.get(record_key)
        if img_path and Path(img_path).exists():
            return str(Path(img_path).resolve())
    return None
def load_records(jsonl_path: str | None=None,image_root: str | None=None)->Dataset:
    resolved_jsonl,resolved_image_root=resolve_data_paths(jsonl_path,image_root)
    path=Path(resolved_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {resolved_jsonl}")
    image_root_path=Path(resolved_image_root)
    if not image_root_path.exists():
        raise FileNotFoundError(f"Image root directory does not exist: {resolved_image_root}")
    raw_records=[]
    with path.open("r",encoding="utf-8") as handle:
        for line_num,line in enumerate(handle,1):
            line=line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {line_num} in {resolved_jsonl}: {e}")
    print(f"Loaded {len(raw_records)} raw records from {resolved_jsonl}")
    examples=[]
    missing_images=[]
    missing_fields=[]
    for idx,record in enumerate(raw_records):
        if not isinstance(record,dict):
            continue
        system_prompt=record.get("system_prompt") or record.get("system") or ""
        prompt_text=record.get("prompt_text") or record.get("prompt") or record.get("html") or record.get("html_text")
        target_text=record.get("target") or record.get("action") or record.get("target_action") or record.get("label")
        if not prompt_text or target_text is None:
            missing_fields.append(idx)
            continue
        resolved_image=_resolve_image_for_record(record,image_root_path)
        if resolved_image is None:
            ds_idx=record.get("dataset_index",f"record_{idx}")
            missing_images.append(ds_idx)
            continue
        examples.append(
            {
                "image": resolved_image,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "target": target_text,
            }
        )
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} records have missing screenshots. "
            f"First 10 dataset_index values: {missing_images[:10]}. "
            f"Expected images at: {resolved_image_root}/{ dataset_index} .jpg"
        )
    if missing_fields:
        print(f"WARNING: {len(missing_fields)} records skipped due to missing prompt_text or target fields.")
    if not examples:
        raise ValueError(f"No usable records found in {resolved_jsonl}")
    print(f"Built dataset with {len(examples)} examples (images verified on disk)")
    return Dataset.from_list(examples)
