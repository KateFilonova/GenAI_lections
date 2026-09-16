# tests/test_audio_info.py

import pytest
from llm_agent.core_v2 import LLMAgent


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать при быстрой проверке

@pytest.mark.integration
def test_audio_info_query_live():
    """Реальный запуск агента для проверки извлечения метаданных MP3-файла."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    query = "Извлеки метаданные из аудиофайла test_audio/sample.mp3 и напиши его длительность и битрейт."

    response = agent.process_query(query)

    # Проверяем, что в ответе присутствуют ключевые поля метаданных
    assert "битрейт" in response.lower() or "bitrate" in response.lower()
    # Проверяем, что модель упомянула длительность
    assert "длитель" in response.lower() or "duration" in response.lower()


@pytest.mark.integration
def test_audio_info_wav_query_live():
    """Реальный запуск агента для проверки извлечения метаданных WAV-файла."""

    agent = LLMAgent(local=True, ollama_model="qwen3:0.6b")
    query = "Какие параметры у аудиофайла test_audio/sample.wav? Сколько там каналов?"

    response = agent.process_query(query)

    # Проверяем, что в ответе упоминаются каналы
    assert "канал" in response.lower() or "channel" in response.lower()


@pytest.mark.integration
def test_audio_info_tags_query_live():
    """Реальный запуск агента для проверки извлечения тегов (исполнитель, альбом)."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    query = "Какой исполнитель и альбом у файла test_audio/track.mp3?"

    response = agent.process_query(query)

    # Проверяем, что в ответе присутствует упоминание исполнителя или альбома
    assert (
        "исполнител" in response.lower()
        or "artist" in response.lower()
        or "альбом" in response.lower()
        or "album" in response.lower()
    )
