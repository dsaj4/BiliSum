from pathlib import Path

import pytest

from video_sum_core.models.tasks import InputType, TaskInput, TaskResult, TaskStatus
from video_sum_core.note_modes import (
    UnknownNoteModeError,
    build_detailed_record_fallback,
    normalize_detailed_record_payload,
    normalize_note_modes,
    render_detailed_record_markdown,
)
from video_sum_core.errors import VideoSumError
from video_sum_core.pipeline.real import PipelineSettings, RealPipelineRunner
from video_sum_core.pipeline.base import PipelineContext
from video_sum_infra.config import (
    ServiceSettings,
    normalize_transcription_provider,
    recommend_mindmap_concurrency,
    recommend_task_concurrency,
)
from video_sum_infra.runtime import default_data_dir
from video_sum_service.settings_manager import SettingsManager, SettingsUpdatePayload


def test_task_input_defaults() -> None:
    task_input = TaskInput(input_type=InputType.URL, source="https://example.com/video")

    assert task_input.options.language == "zh"
    assert "json" in task_input.options.export_formats
    assert task_input.options.note_modes == ["knowledge_note"]


def test_note_modes_normalize_and_reject_unknown_values() -> None:
    assert normalize_note_modes("knowledge-note,detailed-record") == ["knowledge_note", "detailed_record"]
    with pytest.raises(UnknownNoteModeError):
        normalize_note_modes(["missing_mode"])


def test_task_status_values_stable() -> None:
    assert TaskStatus.QUEUED.value == "queued"
    assert TaskStatus.COMPLETED.value == "completed"


def test_pipeline_context_requires_task_id() -> None:
    context = PipelineContext(
        task_id="task-1",
        task_input=TaskInput(input_type=InputType.URL, source="https://example.com/video"),
    )

    assert context.task_id == "task-1"


def test_service_settings_resolve_cuda_runtime() -> None:
    settings = ServiceSettings(
        device_preference="cuda",
        compute_type="auto",
        model_mode="fixed",
        fixed_model="large-v3-turbo",
    )

    model, device, compute_type = settings.resolve_whisper_runtime(cuda_available=True)

    assert model == "large-v3-turbo"
    assert device == "cuda"
    assert compute_type == "float16"


def test_service_settings_normalize_gpu_alias_to_cuda() -> None:
    settings = ServiceSettings(device_preference="gpu", compute_type="int8")

    model, device, compute_type = settings.resolve_whisper_runtime(cuda_available=True)

    assert settings.device_preference == "cuda"
    assert device == "cuda"
    assert compute_type == "int8_float16"


def test_normalize_transcription_provider_aliases() -> None:
    assert normalize_transcription_provider("faster-whisper") == "local"
    assert normalize_transcription_provider("silicon_flow") == "siliconflow"


def test_service_settings_supports_siliconflow_asr_defaults() -> None:
    settings = ServiceSettings(transcription_provider="silicon-flow")

    assert settings.transcription_provider == "siliconflow"
    assert settings.siliconflow_asr_base_url == "https://api.siliconflow.cn/v1"
    assert settings.siliconflow_asr_model == "TeleAI/TeleSpeechASR"


def test_service_settings_default_transcription_provider_is_siliconflow() -> None:
    settings = ServiceSettings()

    assert settings.transcription_provider == "siliconflow"


def test_recommend_task_concurrency_prefers_local_asr_serial_execution() -> None:
    settings = ServiceSettings(transcription_provider="local", runtime_channel="gpu-cu128", device_preference="cuda")

    assert recommend_task_concurrency(settings, cuda_available=True) == 1


def test_recommend_task_concurrency_prefers_cloud_or_gpu_parallel_execution() -> None:
    settings = ServiceSettings(transcription_provider="siliconflow", runtime_channel="base", device_preference="cpu")

    assert recommend_task_concurrency(settings, cuda_available=False) == 2
    assert recommend_mindmap_concurrency() == 1


def test_settings_manager_preserves_existing_local_provider(tmp_path: Path) -> None:
    base_settings = ServiceSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        tasks_dir=tmp_path / "tasks",
    )
    manager = SettingsManager(base_settings)
    settings_path = base_settings.data_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text('{"transcription_provider":"local","fixed_model":"tiny"}', encoding="utf-8")

    loaded = manager.load()

    assert loaded.transcription_provider == "local"
    assert loaded.fixed_model == "tiny"


def test_settings_manager_reports_persisted_file_state(tmp_path: Path) -> None:
    base_settings = ServiceSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        tasks_dir=tmp_path / "tasks",
    )
    manager = SettingsManager(base_settings)

    assert manager.has_persisted_settings is False

    manager.save(SettingsUpdatePayload(llm_enabled=True))

    assert manager.has_persisted_settings is True


def test_settings_manager_migrates_missing_task_concurrency_fields(tmp_path: Path) -> None:
    base_settings = ServiceSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        tasks_dir=tmp_path / "tasks",
    )
    manager = SettingsManager(base_settings)
    settings_path = base_settings.data_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text('{"transcription_provider":"local","fixed_model":"tiny"}', encoding="utf-8")

    loaded = manager.load()
    persisted = ServiceSettings.model_validate_json(settings_path.read_text(encoding="utf-8"))

    assert loaded.task_concurrency == 1
    assert loaded.mindmap_concurrency == 1
    assert persisted.task_concurrency == 1
    assert persisted.mindmap_concurrency == 1


def test_settings_manager_migrates_missing_knowledge_note_prompt_fields(tmp_path: Path) -> None:
    base_settings = ServiceSettings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        tasks_dir=tmp_path / "tasks",
    )
    manager = SettingsManager(base_settings)
    settings_path = base_settings.data_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text('{"summary_mode":"llm"}', encoding="utf-8")

    loaded = manager.load()

    assert loaded.knowledge_note_system_prompt == base_settings.knowledge_note_system_prompt
    assert loaded.knowledge_note_user_prompt_template == base_settings.knowledge_note_user_prompt_template


def test_service_settings_default_to_managed_user_data_dir() -> None:
    settings = ServiceSettings()

    assert settings.data_dir == default_data_dir()
    assert settings.cache_dir == default_data_dir() / "cache"
    assert settings.tasks_dir == default_data_dir() / "tasks"
    assert settings.runtime_channel == "base"


def test_transcription_command_uses_managed_runtime_python(monkeypatch) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path("."), runtime_channel="base"))

    monkeypatch.setattr("video_sum_core.pipeline.real.runtime_python_executable", lambda channel: Path("C:/runtime/python.exe"))

    command = runner._build_transcription_command(
        audio_path=Path("audio.mp3"),
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        progress_path=Path("progress.jsonl"),
        output_path=Path("result.json"),
    )

    assert Path(command[0]) == Path("C:/runtime/python.exe")
    assert command[1:3] == ["-m", "video_sum_core.transcribe_subprocess"]
    assert "--audio-path" in command


def test_transcription_command_requires_managed_runtime_python(monkeypatch) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path("."), runtime_channel="base"))

    monkeypatch.setattr("video_sum_core.pipeline.real.runtime_python_executable", lambda channel: None)

    with pytest.raises(VideoSumError, match="Managed runtime python is unavailable"):
        runner._build_transcription_command(
            audio_path=Path("audio.mp3"),
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            progress_path=Path("progress.jsonl"),
            output_path=Path("result.json"),
        )


def test_real_pipeline_normalizes_empty_llm_summary() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    transcript = "[00:00] 第一条信息\n[00:10] 第二条信息\n[00:20] 第三条信息"
    segments = [
        {"start": 0.0, "end": 5.0, "text": "第一段内容"},
        {"start": 10.0, "end": 15.0, "text": "第二段内容"},
        {"start": 20.0, "end": 25.0, "text": "第三段内容"},
        {"start": 30.0, "end": 35.0, "text": "第四段内容"},
    ]

    summary = runner._normalize_summary(
        {
            "title": "测试视频",
            "overview": "这是一个测试概览。",
            "bulletPoints": [],
            "chapters": [],
        },
        transcript,
        segments,
        "测试视频",
    )

    assert summary["overview"] == "这是一个测试概览。"
    assert len(summary["bulletPoints"]) >= 1
    assert len(summary["chapters"]) >= 1


def test_real_pipeline_builds_strict_summary_prompt() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    messages = runner._build_summary_messages(
        "测试视频",
        "这是一段转写。",
        '[{"start":0,"text":"第一段"}]',
    )

    assert len(messages) == 2
    assert "json" in messages[0]["content"].lower() or "json" in messages[1]["content"].lower()
    assert "不能返回空数组" in messages[1]["content"]
    assert "bulletPoints 必须是 5 到 8 条中文要点" in messages[1]["content"]
    assert "chapters 必须按内容自然分布生成" in messages[1]["content"]
    assert "章节数量不要预设固定值" in messages[1]["content"]
    assert "禁止使用“大章节1”" in messages[1]["content"]
    assert "chapterGroups" in messages[1]["content"]
    assert "每条都要能单独成为一张知识卡片" in messages[1]["content"]
    assert "knowledgeNoteMarkdown" not in messages[1]["content"]


def test_real_pipeline_uses_custom_prompt_template() -> None:
    runner = RealPipelineRunner(
        PipelineSettings(
            tasks_dir=Path("."),
            summary_system_prompt="自定义系统提示词",
            summary_user_prompt_template="标题={title}\n转写={transcript_excerpt}\n分段={segments_excerpt}",
        )
    )

    messages = runner._build_summary_messages("测试标题", "转写内容", "分段内容")

    assert messages[0]["content"] == "自定义系统提示词"
    assert "标题=测试标题" in messages[1]["content"]
    assert "转写=转写内容" in messages[1]["content"]
    assert "分段=分段内容" in messages[1]["content"]


def test_real_pipeline_uses_custom_knowledge_note_prompt_template() -> None:
    runner = RealPipelineRunner(
        PipelineSettings(
            tasks_dir=Path("."),
            knowledge_note_system_prompt="自定义知识笔记系统词",
            knowledge_note_user_prompt_template="标题={title}\n摘要={summary_json}\n正文={transcript_excerpt}",
        )
    )

    messages = runner._build_knowledge_note_messages("测试标题", "转写内容", "分段内容", '{"overview":"概览"}')

    assert messages[0]["content"] == "自定义知识笔记系统词"
    assert "标题=测试标题" in messages[1]["content"]
    assert '摘要={"overview":"概览"}' in messages[1]["content"]
    assert "正文=转写内容" in messages[1]["content"]


def test_real_pipeline_builds_separate_knowledge_note_prompt() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    messages = runner._build_knowledge_note_messages(
        "测试标题",
        "转写内容",
        "分段内容",
        '{"overview":"概览","bulletPoints":["要点一"]}',
    )

    assert len(messages) == 2
    assert "knowledgeNoteMarkdown" in messages[1]["content"]
    assert "不要只是把 bulletPoints 改写一遍" in messages[1]["content"]
    assert "已有结构化摘要" in messages[1]["content"]


def test_llm_payload_forces_json_keyword_when_custom_prompts_miss_it() -> None:
    runner = RealPipelineRunner(
        PipelineSettings(
            tasks_dir=Path("."),
            summary_system_prompt="你是摘要助手。",
            summary_user_prompt_template="标题={title}\n正文={transcript_excerpt}",
        )
    )

    payload = runner._build_llm_summary_payload("测试标题", "转写内容", "分段内容")
    contents = "\n".join(str(message.get("content") or "") for message in payload["messages"])

    assert "json" in contents.lower()


def test_aggregate_series_prompt_uses_page_anchors_instead_of_timestamps() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    payload = runner._build_aggregate_series_summary_payload(
        "测试合集｜全集总结",
        "## P1 开场\n[重点-概览] 开场摘要",
        '[{"page":1,"label":"P1 开场"}]',
    )
    contents = "\n".join(str(message.get("content") or "") for message in payload["messages"])

    assert "全集总结" in contents
    assert "start 必须是该主题最适合跳转的分 P 数字" in contents
    assert "禁止使用时间戳" in contents
    assert "每个分 P 至少保留一个高价值信息点" in contents


def test_knowledge_note_payload_forces_json_keyword() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    payload = runner._build_llm_knowledge_note_payload("测试标题", "转写内容", "分段内容", '{"overview":"概览"}')
    contents = "\n".join(str(message.get("content") or "") for message in payload["messages"])

    assert "json" in contents.lower()


def test_detailed_record_prompt_requests_faithful_transcript_sections() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    payload = runner._build_llm_detailed_record_payload(
        "测试标题",
        "[00:00] 我最近做了一个学习项目。",
        '[{"index":1,"start":0,"end":5,"text":"我最近做了一个学习项目。"}]',
        '{"overview":"概览"}',
    )
    contents = "\n".join(str(message.get("content") or "") for message in payload["messages"])

    assert "sections" in contents
    assert "microSegments" in contents
    assert "不要总结" in contents
    assert "不要把“我”改成“作者/UP 主”" in contents
    assert "忠实保留原始材料的表达方式" in contents
    assert "segments、transitions、compactSummary" not in contents


def test_detailed_record_v2_normalizes_and_renders_transcript_micro_segments() -> None:
    record = normalize_detailed_record_payload(
        {
            "title": "学习项目演示",
            "sections": [
                {
                    "title": "开场",
                    "start": 0,
                    "end": 18,
                    "sourceSegmentIds": [1, 2],
                    "microSegments": [
                        {
                            "start": 0,
                            "end": 8,
                            "text": "我最近做了一个帮助自己学习的项目，这个项目对我的学习帮助很大。",
                            "sourceSegmentIds": [1],
                            "visualRefs": [],
                        },
                        {
                            "start": 8,
                            "end": 18,
                            "text": "所以我想把这个系统的思路和使用方式简单讲一下。",
                            "sourceSegmentIds": [2],
                            "visualRefs": [],
                        },
                    ],
                }
            ],
            "timeline": [{"start": 0, "end": 18, "title": "开场"}],
        },
        title="学习项目演示",
        summary={"overview": "不要出现在正文"},
        segments=[],
    )

    markdown = render_detailed_record_markdown(record)

    assert record["version"] == 2
    assert record["style"] == "discourse_aligned_transcript"
    assert "sections" in record
    assert "我最近做了一个帮助自己学习的项目" in markdown
    assert "作者" not in markdown
    assert "## 逐句实录" in markdown
    assert "**1.1 00:00 - 00:08**" in markdown


def test_detailed_record_fallback_uses_source_segments_instead_of_summary() -> None:
    record = build_detailed_record_fallback(
        title="学习项目演示",
        summary={"overview": "这里是总结", "chapters": [{"title": "开场", "start": 0.0, "summary": "作者介绍项目"}]},
        segments=[
            {"id": 1, "start": 0.0, "end": 5.0, "text": "我最近做了一个帮助自己学习的项目。"},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "这个项目对我的学习帮助很大。"},
        ],
    )
    markdown = render_detailed_record_markdown(record)

    assert record["version"] == 2
    assert "我最近做了一个帮助自己学习的项目" in markdown
    assert "这个项目对我的学习帮助很大" in markdown
    assert "作者介绍项目" not in markdown


def test_aggregate_series_note_prompt_uses_page_positioning() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    payload = runner._build_llm_knowledge_note_payload(
        "测试合集｜全集总结",
        "## P1 开场\n[重点-概览] 开场摘要",
        '[{"page":1,"label":"P1 开场"}]',
        '{"overview":"概览"}',
        source_kind="aggregate_series",
    )
    contents = "\n".join(str(message.get("content") or "") for message in payload["messages"])

    assert "所有定位都使用 P 数" in contents
    assert "不要写时间戳" in contents
    assert "回看路径" in contents


def test_export_result_preserves_llm_usage(tmp_path: Path) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    task_dir = tmp_path / "tmp_export_result"
    task_dir.mkdir(parents=True, exist_ok=True)

    result = runner._export_result(
        task_dir=task_dir,
        title="测试标题",
        transcript="转写内容",
        segments=[{"start": 0.0, "end": 1.0, "text": "第一段"}],
        summary={
            "overview": "概览",
            "bulletPoints": ["要点一"],
            "chapters": [{"title": "章节 1", "start": 0.0, "summary": "章节摘要"}],
            "chapterGroups": [{"title": "大章节 1", "start": 0.0, "summary": "分组摘要", "children": [{"title": "章节 1", "start": 0.0, "summary": "章节摘要"}]}],
            "knowledgeNoteMarkdown": "# 测试标题\n\n## 核心概览\n\n概览",
            "llm_prompt_tokens": 123,
            "llm_completion_tokens": 456,
            "llm_total_tokens": 579,
            "llmDiagnostics": {
                "enabled": True,
                "used": True,
                "provider": "openai-compatible",
                "model": "deepseek-test",
                "fallback_reason": "",
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579,
            },
            "transcriptSource": {
                "provider": "bilibili-subtitle",
                "source": "dm_view",
                "lan": "ai-zh",
                "lan_doc": "AI Chinese",
                "is_ai": True,
                "url_host": "i0.hdslb.com",
            },
        },
    )

    assert result.llm_prompt_tokens == 123
    assert result.llm_completion_tokens == 456
    assert result.llm_total_tokens == 579
    assert result.knowledge_note_markdown.startswith("# 测试标题")
    assert result.chapter_groups[0]["children"][0]["title"] == "章节 1"
    assert result.artifacts["knowledge_note_path"].endswith("knowledge_note.md")
    assert result.artifacts["llm_used"] == "true"
    assert result.artifacts["llm_model"] == "deepseek-test"
    assert result.artifacts["subtitle_provider"] == "ai-zh"
    assert "transcript_source_json" in result.artifacts


def test_export_result_writes_detailed_record_variant(tmp_path: Path) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path("."), llm_enabled=False))
    task_dir = tmp_path / "tmp_export_detailed_record"
    task_dir.mkdir(parents=True, exist_ok=True)

    result = runner._export_result(
        task_dir=task_dir,
        title="细粒度笔记",
        transcript="[00:00] 开场\n[00:10] 展开观点",
        segments=[
            {"start": 0.0, "end": 8.0, "text": "开场说明主题"},
            {"start": 10.0, "end": 18.0, "text": "展开核心观点"},
        ],
        summary={
            "overview": "概述",
            "bulletPoints": ["要点一"],
            "chapters": [{"title": "开场", "start": 0.0, "summary": "说明主题"}],
            "knowledgeNoteMarkdown": "# 知识笔记",
        },
        note_modes=["knowledge_note", "detailed_record"],
        primary_note_mode="detailed_record",
    )

    assert result.primary_note_mode == "detailed_record"
    assert {variant.id for variant in result.note_variants} == {"knowledge_note", "detailed_record"}
    detailed = next(variant for variant in result.note_variants if variant.id == "detailed_record")
    assert detailed.structured_artifact_path is not None
    assert detailed.artifact_path is not None
    assert Path(detailed.structured_artifact_path).exists()
    assert Path(detailed.artifact_path).exists()
    assert result.artifacts["note_variant_detailed_record_json_path"].endswith("detailed_record.json")


def test_visual_base_note_uses_primary_detailed_record_variant() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    result = TaskResult(
        knowledge_note_markdown="# 知识笔记\n\n这是知识笔记。",
        primary_note_mode="detailed_record",
        note_variants=[
            {
                "id": "detailed_record",
                "label": "逐句实录",
                "markdown": "# 逐句实录\n\n我最近做了一个项目。",
            }
        ],
    )

    base_note = runner._visual_base_note(result)

    assert base_note["mode"] == "detailed_record"
    assert base_note["label"] == "逐句实录"
    assert "我最近做了一个项目" in str(base_note["markdown"])


def test_visual_context_records_note_mode(tmp_path: Path) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    context = runner._build_visual_context_payload(
        task_id="task-1",
        status="ready",
        source_kind="video",
        frames=[],
        observations=[],
        warnings=[],
        note_path=tmp_path / "visual_note.md",
        enhanced_note_path=tmp_path / "visual_enhanced_note.md",
        frame_index_path=tmp_path / "frame_index.json",
        keyframe_plan_path=tmp_path / "visual_keyframe_plan.json",
        insert_plan_path=tmp_path / "visual_insert_plan.json",
        mode="frame_insert",
        note_mode="detailed_record",
        note_label="逐句实录",
        insert_count=1,
    )

    assert context["note_mode"] == "detailed_record"
    assert context["note_label"] == "逐句实录"


def test_real_pipeline_normalizes_mindmap_payload_and_repairs_leaf_time() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    result = TaskResult(
        overview="讲解函数概念与例子。",
        knowledge_note_markdown="# 笔记",
        timeline=[
            {"title": "函数定义", "start": 12.0, "summary": "讲函数的定义域和值域。"},
            {"title": "典型例子", "start": 88.0, "summary": "分析二次函数与绝对值函数。"},
        ],
        chapter_groups=[
            {"title": "基础定义", "start": 12.0, "summary": "函数的基本定义。", "children": [{"title": "函数定义", "start": 12.0, "summary": "讲函数的定义域和值域。"}]},
        ],
    )

    mindmap = runner._normalize_mindmap_payload(
        {
            "title": "函数导图",
            "root": "root",
            "nodes": [
                {
                    "id": "root",
                    "label": "函数",
                    "type": "root",
                    "summary": "整体导图",
                    "children": [
                        {
                            "label": "主题1",
                            "type": "theme",
                            "summary": "函数定义",
                            "children": [
                                {
                                    "label": "函数定义",
                                    "type": "leaf",
                                    "summary": "解释定义域和值域",
                                    "children": [],
                                    "source_chapter_titles": ["函数定义"],
                                    "source_chapter_starts": [12.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        title="函数导图",
        result=result,
    )

    assert mindmap.root == "root"
    assert mindmap.nodes[0].children[0].type == "theme"
    assert mindmap.nodes[0].children[0].label != "主题1"
    assert mindmap.nodes[0].children[0].children[0].type == "leaf"
    assert mindmap.nodes[0].children[0].children[0].time_anchor == 12.0


def test_real_pipeline_preserves_latex_and_allows_richer_mindmap_structure() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    result = TaskResult(
        overview="讲解数列极限的定义、证明和典型例题。",
        knowledge_note_markdown="# 笔记",
        timeline=[
            {"title": "定义", "start": 10.0, "summary": "介绍 $\\varepsilon$-$N$ 定义。"},
            {"title": "例题", "start": 120.0, "summary": "证明 $\\frac{1}{n}\\to 0$。"},
        ],
    )

    mindmap = runner._normalize_mindmap_payload(
        {
            "title": "数列极限导图",
            "root": "root",
            "nodes": [
                {
                    "id": "root",
                    "label": "数列极限",
                    "type": "root",
                    "summary": "整体导图",
                    "children": [
                        {
                            "label": f"主题{index + 1}",
                            "type": "theme",
                            "summary": "围绕公式 $\\frac{1}{n}$ 与 $(-1)^n$ 的讨论，说明如何把定义、例题、注意事项一起组织成更完整的知识树。",
                            "children": [
                                {
                                    "label": "例题：$\\frac{1}{n}\\to 0$",
                                    "type": "leaf",
                                    "summary": "利用 $N>\\frac{1}{\\varepsilon}$ 说明当 $n>N$ 时有 $\\left|\\frac{1}{n}-0\\right|<\\varepsilon$。",
                                    "children": [],
                                    "source_chapter_titles": ["例题"],
                                    "source_chapter_starts": [120.0],
                                }
                            ],
                        }
                        for index in range(7)
                    ],
                }
            ],
        },
        title="数列极限导图",
        result=result,
    )

    assert len(mindmap.nodes[0].children) == 7
    assert "\\frac{1}{n}" in mindmap.nodes[0].children[0].children[0].label
    assert "\\left|\\frac{1}{n}-0\\right|<\\varepsilon" in mindmap.nodes[0].children[0].children[0].summary


def test_clean_title_drops_formula_when_truncation_would_split_it() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    cleaned = runner._clean_title(
        "核心公式 $f(x_0+\\Delta x) \\approx f(x_0)+f'(x_0)\\Delta x$ 的几何解释与近似推导"
    )

    assert cleaned.count("$") % 2 == 0
    assert cleaned == "核心公式"


def test_render_user_prompt_template_preserves_latex_braces() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    rendered = runner._render_user_prompt_template(
        "标题：{title}\n公式：$\\frac{1}{n}$\n示例：{{\"x\":1}}\n摘要：{summary_json}",
        title="数列极限",
        transcript_excerpt="",
        segments_excerpt="",
        summary_json='{"ok": true}',
    )

    assert "$\\frac{1}{n}$" in rendered
    assert '{"x":1}' in rendered
    assert "数列极限" in rendered


def test_build_summary_chunks_splits_large_segments() -> None:
    runner = RealPipelineRunner(
        PipelineSettings(tasks_dir=Path("."), summary_chunk_target_chars=80, summary_chunk_overlap_segments=1)
    )
    segments = [
        {
            "start": float(index * 10),
            "text": f"这是第{index}段内容，用来测试分块摘要是否会自动切分。" * 12,
        }
        for index in range(10)
    ]

    chunks = runner._build_summary_chunks(segments)

    assert len(chunks) > 1
    assert all(chunk["transcript"] for chunk in chunks)
    assert all(chunk["segments_json"] for chunk in chunks)


def test_build_aggregate_summary_inputs_collects_partial_summaries() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    merged_chapters = [
        {"title": "全局章节 A", "start": 10.0, "summary": "全局章节摘要 A"},
        {"title": "全局章节 B", "start": 20.0, "summary": "全局章节摘要 B"},
    ]

    transcript, segments_json = runner._build_aggregate_summary_inputs(
        [
            {
                "chunk_index": 1,
                "title": "分块一",
                "overview": "这里是第一块概览。",
                "bulletPoints": ["要点一", "要点二"],
                "chapters": [{"title": "章节 A", "start": 10.0, "summary": "章节摘要 A"}],
            },
            {
                "chunk_index": 2,
                "title": "分块二",
                "overview": "这里是第二块概览。",
                "bulletPoints": ["要点三"],
                "chapters": [{"title": "章节 B", "start": 20.0, "summary": "章节摘要 B"}],
            },
        ],
        merged_chapters,
    )

    assert "合并后的全局章节" in transcript
    assert "全局章节 A" in transcript
    assert "分块 1" in transcript
    assert "章节 A" in transcript
    assert "章节摘要 B" in transcript
    assert "全局章节 A" in segments_json


def test_parse_llm_json_content_accepts_fenced_json() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    parsed = runner._parse_llm_json_content(
        '```json\n{"title":"测试","overview":"概览","bulletPoints":[],"chapters":[]}\n```'
    )

    assert parsed["title"] == "测试"


def test_parse_llm_json_content_accepts_control_characters_with_strict_false() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    parsed = runner._parse_llm_json_content(
        '{"title":"测试","overview":"第一行\n第二行","bulletPoints":[],"chapters":[]}'
    )

    assert "第一行" in parsed["overview"]


def test_real_pipeline_builds_knowledge_note_markdown_fallback() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    summary = runner._normalize_summary(
        {
            "title": "高数笔记",
            "overview": "这里解释极限的核心思路。",
            "bulletPoints": ["先固定变量，再讨论趋近过程。"],
            "chapters": [{"title": "定义", "start": 0.0, "summary": "介绍极限定义与阅读方式。"}],
        },
        "第一行转写\n第二行转写",
        [{"start": 0.0, "end": 1.0, "text": "介绍极限定义"}],
        "高数笔记",
    )

    assert summary["knowledgeNoteMarkdown"]
    assert "## 核心概览" in str(summary["knowledgeNoteMarkdown"])
    assert "### 定义" in str(summary["knowledgeNoteMarkdown"])
    assert summary["chapterGroups"]
    assert str(summary["chapterGroups"][0]["title"]) == "定义"


def test_real_pipeline_normalizes_placeholder_titles_from_summary_text() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))

    summary = runner._normalize_summary(
        {
            "title": "高数笔记",
            "overview": "这里解释极限的核心思路。",
            "bulletPoints": ["先固定变量，再讨论趋近过程。"],
            "chapters": [
                {"title": "章节 1", "start": 0.0, "summary": "函数定义与符号规范，解释映射、定义域和值域。"},
                {"title": "章节 2", "start": 120.0, "summary": "典型函数示例分析，对比绝对值函数和符号函数。"},
            ],
            "chapterGroups": [
                {
                    "title": "大章节 1",
                    "start": 0.0,
                    "summary": "函数定义与符号规范；典型函数示例分析",
                    "children": [
                        {"title": "章节 1", "start": 0.0, "summary": "函数定义与符号规范，解释映射、定义域和值域。"},
                        {"title": "章节 2", "start": 120.0, "summary": "典型函数示例分析，对比绝对值函数和符号函数。"},
                    ],
                }
            ],
        },
        "第一行转写\n第二行转写",
        [{"start": 0.0, "end": 1.0, "text": "介绍函数定义"}],
        "高数笔记",
    )

    assert str(summary["chapters"][0]["title"]).startswith("函数定义与符号规范")
    assert str(summary["chapterGroups"][0]["title"]).startswith("函数定义与符号规范")


def test_real_pipeline_allows_more_chapters_for_long_content() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    segments = [
        {
            "start": float(index * 360),
            "end": float(index * 360 + 40),
            "text": f"第{index + 1}段：这是一个新的主题，包含定义、例子和结论，确保会形成独立章节。",
        }
        for index in range(12)
    ]

    chapters = runner._build_chapters_fallback(segments)
    groups = runner._build_chapter_groups_from_chapters(chapters)

    assert len(chapters) >= 8
    assert len(groups) >= 3


def test_real_pipeline_merges_partial_chapters_without_recompressing() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    segments = [
        {"start": float(index * 300), "end": float(index * 300 + 30), "text": f"第{index + 1}段主题内容"}
        for index in range(8)
    ]

    merged_chapters = runner._merge_partial_chapters(
        [
            {
                "chunk_index": 1,
                "chunk_start": 0.0,
                "chunk_end": 930.0,
                "chapters": [
                    {"title": "章节 1", "start": 0.0, "summary": "函数定义与映射关系"},
                    {"title": "章节 2", "start": 300.0, "summary": "定义域和值域分析"},
                    {"title": "章节 3", "start": 600.0, "summary": "典型函数例子"},
                    {"title": "章节 4", "start": 900.0, "summary": "符号函数与取整函数"},
                ],
            },
            {
                "chunk_index": 2,
                "chunk_start": 600.0,
                "chunk_end": 2130.0,
                "chapters": [
                    {"title": "章节 3", "start": 600.0, "summary": "典型函数例子"},
                    {"title": "章节 4", "start": 900.0, "summary": "符号函数与取整函数"},
                    {"title": "章节 5", "start": 1200.0, "summary": "有界性的定义"},
                    {"title": "章节 6", "start": 1500.0, "summary": "无界性判定"},
                    {"title": "章节 7", "start": 1800.0, "summary": "单调性与区间指定"},
                    {"title": "章节 8", "start": 2100.0, "summary": "课程小结"},
                ],
            },
        ],
        segments,
    )

    assert len(merged_chapters) >= 8
    assert merged_chapters[0]["title"] == "函数定义与映射关系"
    assert merged_chapters[-1]["title"] == "课程小结"


def test_real_pipeline_prefers_structural_chapters_when_aggregate_llm_is_too_short() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    segments = [
        {"start": float(index * 300), "end": float(index * 300 + 30), "text": f"第{index + 1}段主题内容"}
        for index in range(8)
    ]
    merged_chapters = [
        {"title": f"主题 {index + 1}", "start": float(index * 300), "summary": f"第{index + 1}个章节摘要"}
        for index in range(8)
    ]

    result = runner._merge_structured_summary(
        merged={
            "title": "聚合标题",
            "overview": "聚合概览",
            "bulletPoints": ["要点一", "要点二"],
            "chapters": [
                {"title": "压缩章节 1", "start": 0.0, "summary": "压缩摘要 1"},
                {"title": "压缩章节 2", "start": 900.0, "summary": "压缩摘要 2"},
            ],
        },
        partial_summaries=[
            {"chunk_index": 1, "overview": "分块概览一", "bulletPoints": ["分块要点一"]},
            {"chunk_index": 2, "overview": "分块概览二", "bulletPoints": ["分块要点二"]},
        ],
        merged_chapters=merged_chapters,
        transcript="第一行\n第二行",
        segments=segments,
        title="原始标题",
    )

    assert len(result["chapters"]) == 8
    assert len(result["chapterGroups"]) >= 2
    assert len(result["bulletPoints"]) >= 4


def test_real_pipeline_rebalances_long_video_chapters_to_cover_tail() -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=Path(".")))
    segments = [
        {"start": float(index * 180), "end": float(index * 180 + 20), "text": f"第{index + 1}段主题内容"}
        for index in range(24)
    ]
    chapters = [
        {"title": f"章节 {index + 1}", "start": float(index * 180), "summary": f"第{index + 1}个章节摘要"}
        for index in range(24)
    ]

    balanced = runner._rebalance_chapters_for_coverage(chapters, segments)

    assert len(balanced) >= 10
    assert balanced[0]["start"] == 0.0
    assert float(balanced[-1]["start"]) >= 3600.0
