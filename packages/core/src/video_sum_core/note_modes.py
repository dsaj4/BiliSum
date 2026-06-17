from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


KNOWLEDGE_NOTE_MODE = "knowledge_note"
DETAILED_RECORD_MODE = "detailed_record"
DETAILED_RECORD_STYLE = "discourse_aligned_transcript"
DEFAULT_NOTE_MODES = [KNOWLEDGE_NOTE_MODE]


@dataclass(frozen=True)
class NoteModeDefinition:
    id: str
    label: str
    description: str


NOTE_MODE_REGISTRY: dict[str, NoteModeDefinition] = {
    KNOWLEDGE_NOTE_MODE: NoteModeDefinition(
        id=KNOWLEDGE_NOTE_MODE,
        label="知识笔记",
        description="Legacy semantic Markdown knowledge note.",
    ),
    DETAILED_RECORD_MODE: NoteModeDefinition(
        id=DETAILED_RECORD_MODE,
        label="逐句实录",
        description="Discourse-aligned faithful transcript with structured JSON and Markdown.",
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
) -> dict[str, object]:
    chapters = [item for item in summary.get("chapters") or [] if isinstance(item, dict)]
    source_segments = [item for item in segments if isinstance(item, dict)]
    if not source_segments and chapters:
        source_segments = [
            {
                "start": _safe_float(item.get("start")),
                "end": _safe_float(item.get("end")),
                "text": str(item.get("text") or item.get("summary") or item.get("title") or "").strip(),
            }
            for item in chapters
        ]
    sections = _build_fallback_sections(chapters, source_segments)
    return _build_detailed_record_v2(title, sections, {})


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
        }

    segments = [item for item in record.get("segments") or [] if isinstance(item, dict)]
    return {
        "segment_count": len(segments),
        "has_compact_summary": bool(str(record.get("compactSummary") or "").strip()),
        "json_chars": len(json.dumps(record, ensure_ascii=False)),
    }


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
    timeline = [item for item in record.get("timeline") or [] if isinstance(item, dict)]
    if timeline:
        lines.extend(["### 时间轴", ""])
        for item in timeline:
            anchor = _format_anchor(_safe_float(item.get("start")), _safe_float(item.get("end")))
            label = str(item.get("title") or "").strip()
            if anchor and label:
                lines.append(f"- {anchor} {label}")
            elif label:
                lines.append(f"- {label}")
            elif anchor:
                lines.append(f"- {anchor}")
        lines.append("")

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
