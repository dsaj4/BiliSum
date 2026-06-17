from pathlib import Path

from video_sum_core.pipeline.real import PipelineSettings, RealPipelineRunner


def test_local_visual_note_inserts_multiple_images_for_same_heading(tmp_path: Path) -> None:
    runner = RealPipelineRunner(PipelineSettings(tasks_dir=tmp_path, visual_note_mode="frame_insert"))

    note = runner._compose_visual_note_locally(
        "# Project\n\n## Shared chapter\n\nText.\n\n## Other chapter\n\nMore text.",
        {
            "insertions": [
                {"frame_id": "f0001", "markdown_image": "visual://f0001", "alt": "A", "anchor_heading": "Shared chapter"},
                {"frame_id": "f0002", "markdown_image": "visual://f0002", "alt": "B", "anchor_heading": "Shared chapter"},
                {"frame_id": "f0003", "markdown_image": "visual://f0003", "alt": "C", "anchor_heading": "Missing chapter"},
            ]
        },
    )

    assert note.index("![A](visual://f0001)") < note.index("![B](visual://f0002)")
    assert note.index("![B](visual://f0002)") < note.index("Text.")
    assert "## 补充截图" in note
    assert "![C](visual://f0003)" in note
