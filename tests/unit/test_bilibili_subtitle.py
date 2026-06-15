import json
from unittest.mock import MagicMock, patch

import pytest

from video_sum_service.integrations import fetch_bilibili_subtitle


def test_fetch_bilibili_subtitle_success():
    """测试成功获取字幕（通过 WBI API）"""
    # Mock WBI API response
    mock_wbi_response = MagicMock()
    mock_wbi_response.status_code = 200
    mock_wbi_response.json.return_value = {
        "code": 0,
        "data": {
            "subtitle": {
                "subtitles": [
                    {"subtitle_url": "https://i0.hdslb.com/bfs/subtitle/test.json", "lan": "zh-CN", "lan_doc": "中文（简体）"}
                ]
            }
        }
    }

    # Mock subtitle content response
    mock_subtitle_response = MagicMock()
    mock_subtitle_response.status_code = 200
    mock_subtitle_response.json.return_value = {
        "body": [
            {"from": 0.0, "to": 1.5, "content": "第一句"},
            {"from": 1.5, "to": 3.0, "content": "第二句"},
        ]
    }

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        # 第一次调用：WBI API，第二次调用：字幕内容
        mock_instance.get.side_effect = [mock_wbi_response, mock_subtitle_response]

        result = fetch_bilibili_subtitle(aid=12345, cid=67890, cookie="test_cookie", bvid="BV1test")

    assert result is not None
    assert result["transcript"] == "第一句\n第二句"
    assert len(result["segments"]) == 2
    assert result["segments"][0] == {"start": 0.0, "end": 1.5, "text": "第一句"}
    assert result["segments"][1] == {"start": 1.5, "end": 3.0, "text": "第二句"}


def test_fetch_bilibili_subtitle_no_subtitles():
    """测试视频无字幕"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "subtitle": {
                "subtitles": []
            }
        }
    }

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.return_value = mock_response

        result = fetch_bilibili_subtitle(aid=12345, cid=67890)

    assert result is None


def test_fetch_bilibili_subtitle_api_error():
    """测试 API 返回错误状态码"""
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.return_value = mock_response

        result = fetch_bilibili_subtitle(aid=12345, cid=67890)

    assert result is None


def test_fetch_bilibili_subtitle_network_error():
    """测试网络异常"""
    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.side_effect = Exception("Network error")

        result = fetch_bilibili_subtitle(aid=12345, cid=67890)

    assert result is None


def test_fetch_bilibili_subtitle_empty_body():
    """测试字幕 body 为空"""
    mock_player_response = MagicMock()
    mock_player_response.status_code = 200
    mock_player_response.json.return_value = {
        "data": {
            "subtitle": {
                "subtitles": [{"subtitle_url": "https://example.com/subtitle.json"}]
            }
        }
    }

    mock_subtitle_response = MagicMock()
    mock_subtitle_response.status_code = 200
    mock_subtitle_response.json.return_value = {"body": []}

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.side_effect = [mock_player_response, mock_subtitle_response]

        result = fetch_bilibili_subtitle(aid=12345, cid=67890)

    assert result is None


def test_fetch_bilibili_subtitle_with_protocol_fix():
    """测试字幕 URL 缺少协议前缀"""
    # Mock WBI API response with protocol-less URL
    mock_wbi_response = MagicMock()
    mock_wbi_response.status_code = 200
    mock_wbi_response.json.return_value = {
        "code": 0,
        "data": {
            "subtitle": {
                "subtitles": [{"subtitle_url": "//i0.hdslb.com/subtitle.json", "lan": "zh-CN", "lan_doc": "中文（简体）"}]
            }
        }
    }

    # Mock subtitle content response
    mock_subtitle_response = MagicMock()
    mock_subtitle_response.status_code = 200
    mock_subtitle_response.json.return_value = {
        "body": [{"from": 0.0, "to": 1.0, "content": "测试"}]
    }

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.side_effect = [mock_wbi_response, mock_subtitle_response]

        result = fetch_bilibili_subtitle(aid=12345, cid=67890, bvid="BV1test")

    assert result is not None
    # 验证第二次请求的 URL 包含 https:
    second_call = mock_instance.get.call_args_list[1]
    assert second_call[0][0].startswith("https://")


def test_fetch_bilibili_subtitle_falls_back_to_dm_view_ai_subtitle():
    mock_wbi_response = MagicMock()
    mock_wbi_response.status_code = 200
    mock_wbi_response.json.return_value = {
        "code": 0,
        "data": {
            "subtitle": {
                "subtitles": []
            }
        }
    }

    mock_dm_response = MagicMock()
    mock_dm_response.status_code = 200
    mock_dm_response.json.return_value = {
        "code": 0,
        "data": {
            "subtitle": {
                "subtitles": [
                    {
                        "subtitle_url": "https://i0.hdslb.com/bfs/subtitle/ai.json",
                        "lan": "ai-zh",
                        "lan_doc": "AI Chinese",
                    }
                ]
            }
        }
    }

    mock_subtitle_response = MagicMock()
    mock_subtitle_response.status_code = 200
    mock_subtitle_response.json.return_value = {
        "body": [
            {"from": 0.0, "to": 2.0, "content": "first ai segment"},
            {"from": 2.0, "to": 4.5, "content": "second ai segment"},
        ]
    }

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.side_effect = [mock_wbi_response, mock_dm_response, mock_subtitle_response]

        result = fetch_bilibili_subtitle(aid=12345, cid=67890, cookie="SESSDATA=test", bvid="BV1test")

    assert result is not None
    assert result["transcript"] == "first ai segment\nsecond ai segment"
    assert result["segments"] == [
        {"start": 0.0, "end": 2.0, "text": "first ai segment"},
        {"start": 2.0, "end": 4.5, "text": "second ai segment"},
    ]

    requested_urls = [call.args[0] for call in mock_instance.get.call_args_list]
    assert requested_urls[0] == "https://api.bilibili.com/x/player/wbi/v2?bvid=BV1test&cid=67890"
    assert requested_urls[1] == "https://api.bilibili.com/x/v2/dm/view?type=1&oid=67890&pid=12345"
    assert requested_urls[2] == "https://i0.hdslb.com/bfs/subtitle/ai.json"
