from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


KNOWLEDGE_NOTE_MODE = "knowledge_note"
DETAILED_RECORD_MODE = "detailed_record"
DETAILED_RECORD_STYLE = "discourse_aligned_transcript"
DEFAULT_NOTE_MODES = [KNOWLEDGE_NOTE_MODE]
MICRO_MIN_CHARS = 100
MICRO_TARGET_CHARS = 220
MICRO_MAX_CHARS = 420
MICRO_SOFT_GAP_SECONDS = 1.2
MICRO_HARD_GAP_SECONDS = 2.5
MICRO_MAX_DURATION_SECONDS = 90
SECTION_TARGET_MICROS = 8
SECTION_MAX_DURATION_SECONDS = 420

FILLER_PATTERNS = (
    r"(?<![\w\u4e00-\u9fff])[呃嗯啊额](?![\w\u4e00-\u9fff])",
    r"^(呃|嗯|啊|额)[，,\s]+",
)
MID_SENTENCE_FILLER_PREFIXES = (
    "这里",
    "那里",
    "这边",
    "那边",
    "现在",
    "然后",
    "所以",
    "就是",
    "这个",
    "那个",
    "其实",
    "可能",
    "比如说",
    "假如",
    "那么",
    "那",
    "是",
    "在",
    "把",
    "给",
    "让",
    "会",
    "要",
    "就",
    "都",
    "还",
    "再",
    "并",
    "和",
    "跟",
    "对",
    "用",
    "按",
)
DETAILED_RECORD_SUMMARY_MARKERS = (
    "作者认为",
    "作者介绍",
    "本段介绍",
    "本段说明",
    "这一部分",
    "视频中提到",
    "UP主认为",
    "UP 主认为",
)

DISCOURSE_BOUNDARY_PREFIXES = (
    "首先",
    "然后",
    "接下来",
    "另外",
    "但是",
    "不过",
    "所以",
    "那么",
    "回到",
    "总结一下",
    "最后",
    "举个例子",
    "换句话说",
)


@dataclass(frozen=True)
class NoteModeDefinition:
    id: str
    label: str
    description: str
    artifact_stem: str
    content_type: str = "markdown"
    has_structured_artifact: bool = False


@dataclass(frozen=True)
class DetailedRecordFormatterConfig:
    micro_min_chars: int = MICRO_MIN_CHARS
    micro_target_chars: int = MICRO_TARGET_CHARS
    micro_max_chars: int = MICRO_MAX_CHARS
    micro_soft_gap_seconds: float = MICRO_SOFT_GAP_SECONDS
    micro_hard_gap_seconds: float = MICRO_HARD_GAP_SECONDS
    micro_max_duration_seconds: float = MICRO_MAX_DURATION_SECONDS

    def normalized(self) -> "DetailedRecordFormatterConfig":
        micro_min_chars = max(40, int(self.micro_min_chars or MICRO_MIN_CHARS))
        micro_target_chars = max(micro_min_chars, int(self.micro_target_chars or MICRO_TARGET_CHARS))
        micro_max_chars = max(micro_target_chars, int(self.micro_max_chars or MICRO_MAX_CHARS))
        return DetailedRecordFormatterConfig(
            micro_min_chars=micro_min_chars,
            micro_target_chars=micro_target_chars,
            micro_max_chars=micro_max_chars,
            micro_soft_gap_seconds=max(0.0, float(self.micro_soft_gap_seconds)),
            micro_hard_gap_seconds=max(0.0, float(self.micro_hard_gap_seconds)),
            micro_max_duration_seconds=max(10.0, float(self.micro_max_duration_seconds)),
        )


NOTE_MODE_REGISTRY: dict[str, NoteModeDefinition] = {
    KNOWLEDGE_NOTE_MODE: NoteModeDefinition(
        id=KNOWLEDGE_NOTE_MODE,
        label="知识笔记",
        description="Legacy semantic Markdown knowledge note.",
        artifact_stem="knowledge_note",
    ),
    DETAILED_RECORD_MODE: NoteModeDefinition(
        id=DETAILED_RECORD_MODE,
        label="逐句实录",
        description="Discourse-aligned faithful transcript with structured JSON and Markdown.",
        artifact_stem="detailed_record",
        content_type="markdown+json",
        has_structured_artifact=True,
    ),
}


class UnknownNoteModeError(ValueError):
    pass


def normalize_note_modes(value: object, *, default: list[str] | None = None) -> list[str]:
    fallback = list(default or DEFAULT_NOTE_MODES)
    if value is None:
        raw_items: list[object] = fallback
    elif isinstance(value, str):
        raw_items = [item for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = fallback

    normalized: list[str] = []
    for item in raw_items:
        mode = str(item or "").strip().lower().replace("-", "_")
        if not mode:
            continue
        if mode not in NOTE_MODE_REGISTRY:
            raise UnknownNoteModeError(f"Unknown note mode: {mode}")
        if mode not in normalized:
            normalized.append(mode)
    return normalized or fallback


def note_mode_definition(mode: str) -> NoteModeDefinition:
    normalized = normalize_note_modes([mode])[0]
    return NOTE_MODE_REGISTRY[normalized]


def note_mode_definitions(modes: list[str]) -> list[NoteModeDefinition]:
    return [note_mode_definition(mode) for mode in normalize_note_modes(modes)]


def normalize_primary_note_mode(note_modes: list[str], primary: str | None = None) -> str:
    modes = normalize_note_modes(note_modes)
    if primary:
        selected = normalize_note_modes([primary])[0]
        if selected in modes:
            return selected
    return modes[0]


def normalize_detailed_record_payload(
    payload: dict[str, Any],
    *,
    title: str,
    summary: dict[str, object],
    segments: list[dict[str, object]],
) -> dict[str, object]:
    record_title = str(payload.get("title") or title or summary.get("title") or "").strip()
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, list):
        normalized_sections: list[dict[str, object]] = []
        for index, item in enumerate(raw_sections, start=1):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_detailed_section(item, index)
            if normalized:
                normalized_sections.append(normalized)
        if normalized_sections:
            return _build_detailed_record_v2(record_title, normalized_sections, payload)

    raw_micro_segments = payload.get("microSegments") or payload.get("micro_segments")
    if isinstance(raw_micro_segments, list):
        micro_segments = [
            normalized
            for index, item in enumerate(raw_micro_segments, start=1)
            if isinstance(item, dict)
            for normalized in [_normalize_micro_segment(item, index)]
            if normalized
        ]
        if micro_segments:
            section = _build_section_from_micro_segments(
                index=1,
                title=str(payload.get("sectionTitle") or payload.get("section_title") or "正文").strip(),
                micro_segments=micro_segments,
            )
            return _build_detailed_record_v2(record_title, [section], payload)

    raw_segments = payload.get("segments")
    source_segments = raw_segments if isinstance(raw_segments, list) else []
    normalized_segments: list[dict[str, object]] = []
    for index, item in enumerate(source_segments, start=1):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_detailed_segment(item, index)
        if normalized:
            normalized_segments.append(normalized)

    if not normalized_segments:
        return build_detailed_record_fallback(title=record_title, summary=summary, segments=segments)

    transitions = []
    for item in payload.get("transitions") or []:
        if not isinstance(item, dict):
            continue
        transitions.append(
            {
                "from": _safe_int(item.get("from")),
                "to": _safe_int(item.get("to")),
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    transitions = [item for item in transitions if item["from"] and item["to"] and item["reason"]]

    compact_summary = str(payload.get("compactSummary") or payload.get("compact_summary") or summary.get("overview") or "").strip()
    return {
        "mode": DETAILED_RECORD_MODE,
        "label": NOTE_MODE_REGISTRY[DETAILED_RECORD_MODE].label,
        "version": 1,
        "title": record_title,
        "segments": normalized_segments,
        "transitions": transitions,
        "compactSummary": compact_summary,
    }


def build_detailed_record_fallback(
    *,
    title: str,
    summary: dict[str, object],
    segments: list[dict[str, object]],
    formatter_config: DetailedRecordFormatterConfig | None = None,
) -> dict[str, object]:
    return build_detailed_record_from_segments(
        title=title,
        summary=summary,
        segments=segments,
        formatter_config=formatter_config,
    )


def build_detailed_record_from_segments(
    *,
    title: str,
    summary: dict[str, object],
    segments: list[dict[str, object]],
    formatter_config: DetailedRecordFormatterConfig | None = None,
) -> dict[str, object]:
    config = (formatter_config or DetailedRecordFormatterConfig()).normalized()
    source_segments = _normalize_source_segments(segments)
    if not source_segments:
        chapters = [item for item in summary.get("chapters") or [] if isinstance(item, dict)]
        source_segments = _normalize_source_segments(
            [
                {
                    "id": index,
                    "start": _safe_float(item.get("start")),
                    "end": _safe_float(item.get("end")),
                    "text": str(item.get("text") or item.get("summary") or item.get("title") or "").strip(),
                }
                for index, item in enumerate(chapters, start=1)
            ]
        )
    sentence_units = _build_sentence_units(source_segments)
    micro_segments = _build_micro_segments_from_sentence_units(sentence_units, config)
    sections = _build_sections_from_micro_segments(micro_segments, summary)
    record = _build_detailed_record_v2(title, sections, {})
    record["rawSegmentCount"] = len(source_segments)
    record["sentenceUnitCount"] = len(sentence_units)
    record["coverage"] = _build_coverage(source_segments, sections)
    record["formatter"] = {
        "name": "formatted_transcript_pipeline",
        "micro_min_chars": config.micro_min_chars,
        "micro_target_chars": config.micro_target_chars,
        "micro_max_chars": config.micro_max_chars,
        "soft_gap_seconds": config.micro_soft_gap_seconds,
        "hard_gap_seconds": config.micro_hard_gap_seconds,
        "max_duration_seconds": config.micro_max_duration_seconds,
    }
    return record


def render_detailed_record_markdown(record: dict[str, object]) -> str:
    if _safe_int(record.get("version")) == 2 or isinstance(record.get("sections"), list):
        return _render_detailed_record_markdown_v2(record)

    title = str(record.get("title") or "逐句实录").strip()
    lines = [f"# {title}", "", "## 逐句实录", ""]
    compact_summary = str(record.get("compactSummary") or "").strip()
    if compact_summary:
        lines.extend(["### 摘要", "", compact_summary, ""])
    for item in record.get("segments") or []:
        if not isinstance(item, dict):
            continue
        index = _safe_int(item.get("index")) or 0
        heading = str(item.get("title") or f"片段 {index}").strip()
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        anchor = _format_anchor(start, end)
        lines.append(f"### {index}. {heading}" if index else f"### {heading}")
        if anchor:
            lines.append("")
            lines.append(f"- 时间：{anchor}")
        role = str(item.get("role") or "").strip()
        if role:
            lines.append(f"- 作用：{role}")
        text = str(item.get("text") or "").strip()
        if text:
            lines.extend(["", text])
        key_points = [str(point).strip() for point in item.get("keyPoints") or [] if str(point).strip()]
        if key_points:
            lines.extend(["", "要点："])
            lines.extend([f"- {point}" for point in key_points])
        visual_refs = [str(ref).strip() for ref in item.get("visualRefs") or [] if str(ref).strip()]
        if visual_refs:
            lines.extend(["", "视觉线索："])
            lines.extend([f"- {ref}" for ref in visual_refs])
        lines.append("")
    transitions = [item for item in record.get("transitions") or [] if isinstance(item, dict)]
    if transitions:
        lines.extend(["## 转场关系", ""])
        for item in transitions:
            lines.append(f"- {item.get('from')} -> {item.get('to')}：{item.get('reason')}")
    return "\n".join(lines).strip() + "\n"


def detailed_record_quality(record: dict[str, object]) -> dict[str, object]:
    if _safe_int(record.get("version")) == 2 or isinstance(record.get("sections"), list):
        sections = [item for item in record.get("sections") or [] if isinstance(item, dict)]
        micro_segments = [
            micro
            for section in sections
            for micro in section.get("microSegments", [])
            if isinstance(micro, dict)
        ]
        return {
            "version": 2,
            "style": str(record.get("style") or DETAILED_RECORD_STYLE),
            "section_count": len(sections),
            "micro_segment_count": len(micro_segments),
            "transcript_chars": sum(len(str(item.get("text") or "")) for item in micro_segments),
            "has_timeline": bool(record.get("timeline")),
            "has_visual_refs": any(item.get("visualRefs") for item in micro_segments),
            "faithfulness_mode": "transcript",
            "coverage": record.get("coverage") if isinstance(record.get("coverage"), dict) else {},
        }

    segments = [item for item in record.get("segments") or [] if isinstance(item, dict)]
    return {
        "segment_count": len(segments),
        "has_compact_summary": bool(str(record.get("compactSummary") or "").strip()),
        "json_chars": len(json.dumps(record, ensure_ascii=False)),
    }


def validate_detailed_record_micro_polish(
    micro_segment: dict[str, object],
    payload: dict[str, object],
    *,
    min_length_ratio: float = 0.65,
) -> tuple[bool, str, dict[str, object]]:
    source_ids = [item for item in micro_segment.get("sourceSegmentIds") or [] if item is not None]
    returned_ids = [item for item in payload.get("sourceSegmentIds") or [] if item is not None]
    if returned_ids != source_ids:
        return False, "coverage_mismatch", {}

    text = _normalize_transcript_text(str(payload.get("text") or ""))
    if not text:
        return False, "empty_text", {}

    local_text = _normalize_transcript_text(str(micro_segment.get("text") or ""))
    if len(local_text) >= 40 and len(text) < int(len(local_text) * min_length_ratio):
        return False, "too_short", {}

    if any(marker in text for marker in DETAILED_RECORD_SUMMARY_MARKERS):
        return False, "summary_marker", {}

    if re.search(r"(^|\n)\s{0,3}#{1,6}\s+", text) or re.search(r"(^|\n)\s*[-*]\s+", text):
        return False, "markdown_structure", {}

    edits = [str(item).strip() for item in payload.get("edits") or [] if str(item).strip()]
    return True, "", {"text": text, "sourceSegmentIds": returned_ids, "edits": edits}


def _normalize_detailed_segment(item: dict[str, Any], index: int) -> dict[str, object] | None:
    text = str(item.get("text") or item.get("summary") or "").strip()
    title = str(item.get("title") or "").strip() or _derive_title(text, f"片段 {index}")
    if not text and not title:
        return None
    start = _safe_float(item.get("start"))
    end = _safe_float(item.get("end"))
    key_points = [str(point).strip() for point in item.get("keyPoints") or item.get("key_points") or [] if str(point).strip()]
    visual_refs = [str(ref).strip() for ref in item.get("visualRefs") or item.get("visual_refs") or [] if str(ref).strip()]
    evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
    normalized_evidence = [entry for entry in evidence if isinstance(entry, dict)]
    return {
        "index": _safe_int(item.get("index")) or index,
        "start": start,
        "end": end,
        "title": title,
        "text": text,
        "role": str(item.get("role") or "development").strip(),
        "keyPoints": key_points,
        "evidence": normalized_evidence,
        "visualRefs": visual_refs,
    }


def _build_detailed_record_v2(
    title: str,
    sections: list[dict[str, object]],
    payload: dict[str, Any],
) -> dict[str, object]:
    timeline = []
    raw_timeline = payload.get("timeline")
    if isinstance(raw_timeline, list):
        for item in raw_timeline:
            if not isinstance(item, dict):
                continue
            start = _safe_float(item.get("start"))
            end = _safe_float(item.get("end"))
            label = str(item.get("title") or item.get("label") or "").strip()
            if start is None and end is None and not label:
                continue
            timeline.append({"start": start, "end": end, "title": label})
    if not timeline:
        timeline = [
            {
                "start": section.get("start"),
                "end": section.get("end"),
                "title": section.get("title"),
            }
            for section in sections
        ]

    return {
        "mode": DETAILED_RECORD_MODE,
        "label": NOTE_MODE_REGISTRY[DETAILED_RECORD_MODE].label,
        "version": 2,
        "style": DETAILED_RECORD_STYLE,
        "title": title,
        "sections": sections,
        "timeline": timeline,
    }


def _normalize_detailed_section(item: dict[str, Any], index: int) -> dict[str, object] | None:
    raw_micro_segments = item.get("microSegments") or item.get("micro_segments") or []
    micro_segments = [
        normalized
        for micro_index, micro in enumerate(raw_micro_segments, start=1)
        if isinstance(micro, dict)
        for normalized in [_normalize_micro_segment(micro, micro_index)]
        if normalized
    ]
    if not micro_segments:
        text = str(item.get("text") or item.get("transcript") or "").strip()
        if text:
            micro_segments = [
                {
                    "id": f"s{index}-m1",
                    "index": 1,
                    "start": _safe_float(item.get("start")),
                    "end": _safe_float(item.get("end")),
                    "text": _normalize_transcript_text(text),
                    "sourceSegmentIds": _normalize_source_ids(item.get("sourceSegmentIds") or item.get("source_segment_ids")),
                    "visualRefs": _normalize_visual_refs(item.get("visualRefs") or item.get("visual_refs")),
                }
            ]
    if not micro_segments:
        return None
    return _build_section_from_micro_segments(
        index=index,
        title=str(item.get("title") or "").strip(),
        micro_segments=micro_segments,
        source_segment_ids=_normalize_source_ids(item.get("sourceSegmentIds") or item.get("source_segment_ids")),
    )


def _normalize_micro_segment(item: dict[str, Any], index: int) -> dict[str, object] | None:
    text = str(item.get("text") or item.get("transcript") or "").strip()
    text = _normalize_transcript_text(text)
    if not text:
        return None
    return {
        "id": str(item.get("id") or f"m{index}").strip(),
        "index": _safe_int(item.get("index")) or index,
        "start": _safe_float(item.get("start")),
        "end": _safe_float(item.get("end")),
        "text": text,
        "sourceSegmentIds": _normalize_source_ids(item.get("sourceSegmentIds") or item.get("source_segment_ids")),
        "visualRefs": _normalize_visual_refs(item.get("visualRefs") or item.get("visual_refs")),
    }


def _build_section_from_micro_segments(
    *,
    index: int,
    title: str,
    micro_segments: list[dict[str, object]],
    source_segment_ids: list[object] | None = None,
) -> dict[str, object]:
    start = next((_safe_float(item.get("start")) for item in micro_segments if _safe_float(item.get("start")) is not None), None)
    end = next((_safe_float(item.get("end")) for item in reversed(micro_segments) if _safe_float(item.get("end")) is not None), None)
    normalized_title = title or _derive_title(str(micro_segments[0].get("text") or ""), f"章节 {index}")
    return {
        "id": f"sec_{index}",
        "index": index,
        "title": normalized_title,
        "start": start,
        "end": end,
        "sourceSegmentIds": source_segment_ids or _collect_source_ids(micro_segments),
        "microSegments": micro_segments,
    }


def _normalize_source_segments(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(segments, start=1):
        if not isinstance(item, dict):
            continue
        text = _normalize_transcript_text(str(item.get("text") or "").strip())
        if not text:
            continue
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        if end is None:
            end = start
        normalized.append(
            {
                "id": item.get("id") or index,
                "index": index,
                "start": start,
                "end": end,
                "text": _clean_transcript_text(text),
            }
        )
    return normalized


def _build_sentence_units(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    buffer_chars = 0
    previous_end: float | None = None

    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _safe_float(item.get("start"))
        gap = _gap_seconds(previous_end, start)
        if buffer and gap is not None and gap >= MICRO_HARD_GAP_SECONDS:
            units.append(_flush_sentence_unit(buffer, len(units) + 1))
            buffer = []
            buffer_chars = 0
        buffer.append(item)
        buffer_chars += len(text)
        previous_end = _safe_float(item.get("end")) or start or previous_end
        if buffer_chars >= 160 or _has_sentence_terminal(text):
            units.append(_flush_sentence_unit(buffer, len(units) + 1))
            buffer = []
            buffer_chars = 0

    if buffer:
        units.append(_flush_sentence_unit(buffer, len(units) + 1))
    return units


def _flush_sentence_unit(buffer: list[dict[str, object]], index: int) -> dict[str, object]:
    text = _join_transcript_parts(str(item.get("text") or "") for item in buffer)
    return {
        "id": f"u{index}",
        "index": index,
        "start": _safe_float(buffer[0].get("start")),
        "end": _safe_float(buffer[-1].get("end")) or _safe_float(buffer[-1].get("start")),
        "text": text,
        "sourceSegmentIds": [item.get("id") for item in buffer if item.get("id") is not None],
    }


def _build_micro_segments_from_sentence_units(
    units: list[dict[str, object]],
    config: DetailedRecordFormatterConfig,
) -> list[dict[str, object]]:
    micro_segments: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    last_boundary_index: int | None = None

    for unit in units:
        if buffer and _should_flush_micro_segment(buffer, unit, config):
            flushed_buffer = buffer
            if _buffer_chars(buffer) > config.micro_max_chars and last_boundary_index is not None and last_boundary_index > 0:
                flushed_buffer = buffer[:last_boundary_index]
                buffer = buffer[last_boundary_index:]
            else:
                buffer = []
            micro_segments.append(_flush_micro_unit_buffer(flushed_buffer, len(micro_segments) + 1))
            last_boundary_index = None
        buffer.append(unit)
        if _is_natural_micro_boundary(unit):
            last_boundary_index = len(buffer)

    if buffer:
        micro_segments.append(_flush_micro_unit_buffer(buffer, len(micro_segments) + 1))
    return _merge_short_micro_segments(micro_segments, config)


def _should_flush_micro_segment(
    buffer: list[dict[str, object]],
    next_unit: dict[str, object],
    config: DetailedRecordFormatterConfig,
) -> bool:
    chars = _buffer_chars(buffer)
    next_chars = len(str(next_unit.get("text") or ""))
    duration = _buffer_duration(buffer)
    gap = _gap_seconds(_safe_float(buffer[-1].get("end")), _safe_float(next_unit.get("start")))
    if chars < config.micro_min_chars:
        return False
    if gap is not None and gap >= config.micro_hard_gap_seconds:
        return True
    if duration >= config.micro_max_duration_seconds:
        return True
    if chars + next_chars > config.micro_max_chars:
        return True
    if chars >= config.micro_target_chars and _is_natural_micro_boundary(buffer[-1]):
        return True
    if chars >= config.micro_target_chars and gap is not None and gap >= config.micro_soft_gap_seconds:
        return True
    return False


def _flush_micro_unit_buffer(buffer: list[dict[str, object]], index: int) -> dict[str, object]:
    text = _join_transcript_parts(str(item.get("text") or "") for item in buffer)
    return {
        "id": f"m{index}",
        "index": index,
        "start": _safe_float(buffer[0].get("start")),
        "end": _safe_float(buffer[-1].get("end")) or _safe_float(buffer[-1].get("start")),
        "text": text,
        "sourceSegmentIds": _collect_ids_from_units(buffer),
        "visualRefs": [],
    }


def _merge_short_micro_segments(
    micro_segments: list[dict[str, object]],
    config: DetailedRecordFormatterConfig,
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for item in micro_segments:
        text = str(item.get("text") or "")
        if merged and len(text) < config.micro_min_chars:
            previous = merged[-1]
            previous["text"] = _join_transcript_parts([str(previous.get("text") or ""), text])
            previous["end"] = item.get("end")
            previous["sourceSegmentIds"] = _merge_ids(previous.get("sourceSegmentIds"), item.get("sourceSegmentIds"))
            continue
        merged.append(dict(item))
    for index, item in enumerate(merged, start=1):
        item["id"] = f"m{index}"
        item["index"] = index
    return merged


def _build_sections_from_micro_segments(
    micro_segments: list[dict[str, object]],
    summary: dict[str, object],
) -> list[dict[str, object]]:
    if not micro_segments:
        return []
    chapter_titles = [
        str(item.get("title") or "").strip()
        for item in summary.get("chapters") or []
        if isinstance(item, dict) and str(item.get("title") or "").strip()
    ]
    sections: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    for micro in micro_segments:
        if buffer and _should_flush_section(buffer, micro):
            sections.append(
                _build_section_from_micro_segments(
                    index=len(sections) + 1,
                    title=_section_title(buffer, chapter_titles, len(sections) + 1),
                    micro_segments=buffer,
                )
            )
            buffer = []
        buffer.append(micro)
    if buffer:
        sections.append(
            _build_section_from_micro_segments(
                index=len(sections) + 1,
                title=_section_title(buffer, chapter_titles, len(sections) + 1),
                micro_segments=buffer,
            )
        )
    return sections


def _should_flush_section(buffer: list[dict[str, object]], next_micro: dict[str, object]) -> bool:
    duration = _safe_float(buffer[-1].get("end") or 0) - (_safe_float(buffer[0].get("start")) or 0)
    gap = _gap_seconds(_safe_float(buffer[-1].get("end")), _safe_float(next_micro.get("start")))
    if gap is not None and gap >= 8 and len(buffer) >= 2:
        return True
    if duration >= SECTION_MAX_DURATION_SECONDS and len(buffer) >= 2:
        return True
    if len(buffer) >= SECTION_TARGET_MICROS:
        return True
    return False


def _section_title(buffer: list[dict[str, object]], chapter_titles: list[str], index: int) -> str:
    if index <= len(chapter_titles):
        return chapter_titles[index - 1]
    return _derive_title(str(buffer[0].get("text") or ""), f"章节 {index}")


def _build_coverage(source_segments: list[dict[str, object]], sections: list[dict[str, object]]) -> dict[str, object]:
    source_ids = [item.get("id") for item in source_segments if item.get("id") is not None]
    covered: list[object] = []
    for section in sections:
        for micro in section.get("microSegments") or []:
            if not isinstance(micro, dict):
                continue
            for source_id in micro.get("sourceSegmentIds") or []:
                if source_id not in covered:
                    covered.append(source_id)
    missing = [source_id for source_id in source_ids if source_id not in covered]
    duplicates = _duplicate_ids(
        source_id
        for section in sections
        for micro in section.get("microSegments", [])
        if isinstance(micro, dict)
        for source_id in micro.get("sourceSegmentIds", [])
    )
    return {
        "sourceSegmentCount": len(source_ids),
        "coveredSegmentCount": len(covered),
        "coverageRatio": round(len(covered) / len(source_ids), 4) if source_ids else 1.0,
        "missingSegmentIds": missing[:50],
        "duplicateSegmentIds": duplicates[:50],
        "maxGapSeconds": _max_covered_gap_seconds(sections),
    }


def _build_fallback_sections(
    chapters: list[dict[str, object]],
    segments: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda item: _safe_float(item.get("start")) or 0)
    if chapters:
        sorted_chapters = sorted(chapters, key=lambda item: _safe_float(item.get("start")) or 0)
        sections: list[dict[str, object]] = []
        for index, chapter in enumerate(sorted_chapters, start=1):
            start = _safe_float(chapter.get("start"))
            next_start = _safe_float(sorted_chapters[index].get("start")) if index < len(sorted_chapters) else None
            chapter_segments = [
                segment
                for segment in sorted_segments
                if _segment_in_range(segment, start, next_start)
            ]
            if not chapter_segments:
                continue
            micro_segments = _build_micro_segments_from_source(chapter_segments)
            if not micro_segments:
                continue
            sections.append(
                _build_section_from_micro_segments(
                    index=len(sections) + 1,
                    title=str(chapter.get("title") or "").strip(),
                    micro_segments=micro_segments,
                )
            )
        if sections:
            return sections

    chunk_size = 6
    sections = []
    for offset in range(0, len(sorted_segments), chunk_size):
        micro_segments = _build_micro_segments_from_source(sorted_segments[offset : offset + chunk_size])
        if not micro_segments:
            continue
        sections.append(
            _build_section_from_micro_segments(
                index=len(sections) + 1,
                title="",
                micro_segments=micro_segments,
            )
        )
    return sections


def _build_micro_segments_from_source(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    micro_segments: list[dict[str, object]] = []
    buffer: list[dict[str, object]] = []
    buffer_chars = 0
    buffer_start: float | None = None
    for item in segments:
        text = _normalize_transcript_text(str(item.get("text") or "").strip())
        if not text:
            continue
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        if buffer and _should_flush_micro_buffer(buffer_chars, buffer_start, start):
            micro_segments.append(_flush_micro_buffer(buffer, len(micro_segments) + 1))
            buffer = []
            buffer_chars = 0
            buffer_start = None
        if buffer_start is None:
            buffer_start = start
        buffer.append({"start": start, "end": end, "text": text, "source_id": item.get("id") or len(micro_segments) + len(buffer) + 1})
        buffer_chars += len(text)
    if buffer:
        micro_segments.append(_flush_micro_buffer(buffer, len(micro_segments) + 1))
    return micro_segments


def _should_flush_micro_buffer(chars: int, buffer_start: float | None, next_start: float | None) -> bool:
    if chars >= 260:
        return True
    if buffer_start is not None and next_start is not None and next_start - buffer_start >= 45:
        return True
    return False


def _flush_micro_buffer(buffer: list[dict[str, object]], index: int) -> dict[str, object]:
    text = " ".join(str(item.get("text") or "").strip() for item in buffer if str(item.get("text") or "").strip())
    return {
        "id": f"m{index}",
        "index": index,
        "start": _safe_float(buffer[0].get("start")),
        "end": _safe_float(buffer[-1].get("end")),
        "text": _normalize_transcript_text(text),
        "sourceSegmentIds": [item.get("source_id") for item in buffer if item.get("source_id") is not None],
        "visualRefs": [],
    }


def _render_detailed_record_markdown_v2(record: dict[str, object]) -> str:
    title = str(record.get("title") or "逐句实录").strip()
    lines = [f"# {title}", "", "## 逐句实录", ""]
    sections = [item for item in record.get("sections") or [] if isinstance(item, dict)]
    for section_index, section in enumerate(sections, start=1):
        heading = str(section.get("title") or f"章节 {section_index}").strip()
        anchor = _format_anchor(_safe_float(section.get("start")), _safe_float(section.get("end")))
        lines.append(f"### {section_index}. {heading}")
        if anchor:
            lines.extend(["", f"- 时间：{anchor}"])
        lines.append("")
        micro_segments = [item for item in section.get("microSegments") or [] if isinstance(item, dict)]
        for micro_index, micro in enumerate(micro_segments, start=1):
            micro_anchor = _format_anchor(_safe_float(micro.get("start")), _safe_float(micro.get("end")))
            if micro_anchor:
                lines.append(f"**{section_index}.{micro_index} {micro_anchor}**")
                lines.append("")
            text = str(micro.get("text") or "").strip()
            if text:
                lines.append(text)
                lines.append("")
            visual_refs = [str(ref).strip() for ref in micro.get("visualRefs") or [] if str(ref).strip()]
            if visual_refs:
                lines.extend([f"![截图]({ref})" for ref in visual_refs])
                lines.append("")
    return "\n".join(lines).strip() + "\n"


def _normalize_transcript_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _clean_transcript_text(text: str) -> str:
    cleaned = _normalize_transcript_text(text)
    cleaned = cleaned.replace(" ,", "，").replace(" .", "。").replace(" ?", "？").replace(" !", "！")
    cleaned = cleaned.replace(",", "，").replace("?", "？").replace("!", "！")
    cleaned = re.sub(r"\s+([，。！？；：])", r"\1", cleaned)
    cleaned = re.sub(r"([，。！？；：])\s+", r"\1", cleaned)
    for pattern in FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = _remove_mid_sentence_fillers(cleaned)
    cleaned = re.sub(r"(然后)[，\s]*(然后)", r"\1", cleaned)
    cleaned = re.sub(r"(就是)[，\s]*(就是)", r"\1", cleaned)
    cleaned = re.sub(r"(这个)[，\s]*(这个)", r"\1", cleaned)
    cleaned = re.sub(r"快速一的进入", "快速地进入", cleaned)
    return _normalize_transcript_text(cleaned)


def _remove_mid_sentence_fillers(text: str) -> str:
    prefixes = "|".join(re.escape(prefix) for prefix in sorted(MID_SENTENCE_FILLER_PREFIXES, key=len, reverse=True))
    cleaned = re.sub(
        rf"({prefixes})[呃嗯啊额](?=[\u4e00-\u9fffA-Za-z0-9])",
        r"\1",
        text,
    )
    return re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])[呃嗯啊额](?=[\u4e00-\u9fffA-Za-z0-9])", "", cleaned)


def _join_transcript_parts(parts: Any) -> str:
    text = "".join(str(part or "").strip() for part in parts if str(part or "").strip())
    text = re.sub(r"([。！？])(?=[\u4e00-\u9fffA-Za-z0-9])", r"\1", text)
    return _clean_transcript_text(text)


def _has_sentence_terminal(text: str) -> bool:
    return str(text or "").rstrip().endswith(("。", "！", "？", ".", "!", "?"))


def _is_natural_micro_boundary(unit: dict[str, object]) -> bool:
    text = str(unit.get("text") or "").strip()
    if _has_sentence_terminal(text):
        return True
    return any(text.startswith(prefix) for prefix in DISCOURSE_BOUNDARY_PREFIXES)


def _buffer_chars(buffer: list[dict[str, object]]) -> int:
    return sum(len(str(item.get("text") or "")) for item in buffer)


def _buffer_duration(buffer: list[dict[str, object]]) -> float:
    if not buffer:
        return 0.0
    start = _safe_float(buffer[0].get("start")) or 0.0
    end = _safe_float(buffer[-1].get("end")) or _safe_float(buffer[-1].get("start")) or start
    return max(0.0, end - start)


def _gap_seconds(previous_end: float | None, next_start: float | None) -> float | None:
    if previous_end is None or next_start is None:
        return None
    return max(0.0, next_start - previous_end)


def _collect_ids_from_units(units: list[dict[str, object]]) -> list[object]:
    ids: list[object] = []
    for unit in units:
        for source_id in unit.get("sourceSegmentIds") or []:
            if source_id not in ids:
                ids.append(source_id)
    return ids


def _merge_ids(left: object, right: object) -> list[object]:
    ids: list[object] = []
    for value in (left, right):
        if not isinstance(value, list):
            continue
        for item in value:
            if item is not None and item not in ids:
                ids.append(item)
    return ids


def _duplicate_ids(values: Any) -> list[object]:
    seen: list[object] = []
    duplicates: list[object] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.append(value)
    return duplicates


def _max_covered_gap_seconds(sections: list[dict[str, object]]) -> float:
    micros = [
        micro
        for section in sections
        for micro in section.get("microSegments", [])
        if isinstance(micro, dict)
    ]
    micros.sort(key=lambda item: _safe_float(item.get("start")) or 0.0)
    max_gap = 0.0
    previous_end: float | None = None
    for micro in micros:
        start = _safe_float(micro.get("start"))
        gap = _gap_seconds(previous_end, start)
        if gap is not None:
            max_gap = max(max_gap, gap)
        previous_end = _safe_float(micro.get("end")) or start or previous_end
    return round(max_gap, 3)


def _normalize_source_ids(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return [item for item in value if item is not None and str(item).strip()]


def _normalize_visual_refs(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _collect_source_ids(micro_segments: list[dict[str, object]]) -> list[object]:
    ids: list[object] = []
    for item in micro_segments:
        for source_id in item.get("sourceSegmentIds") or []:
            if source_id not in ids:
                ids.append(source_id)
    return ids


def _segment_in_range(segment: dict[str, object], start: float | None, end: float | None) -> bool:
    segment_start = _safe_float(segment.get("start")) or 0
    if start is not None and segment_start < start:
        return False
    if end is not None and segment_start >= end:
        return False
    return True


def _derive_title(text: str, fallback: str) -> str:
    normalized = str(text or "").strip().replace("\n", " ")
    if not normalized:
        return fallback
    for sep in ("。", "；", ";", "，", ",", " "):
        if sep in normalized:
            normalized = normalized.split(sep, 1)[0].strip()
            break
    return normalized[:24] or fallback


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_int(value: object) -> int | None:
    try:
        parsed = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed


def _format_anchor(start: float | None, end: float | None) -> str:
    if start is None and end is None:
        return ""
    if end is None or end == start:
        return _format_seconds(start or 0)
    return f"{_format_seconds(start or 0)} - {_format_seconds(end)}"


def _format_seconds(value: float) -> str:
    total = max(0, int(value))
    return f"{total // 60:02d}:{total % 60:02d}"
