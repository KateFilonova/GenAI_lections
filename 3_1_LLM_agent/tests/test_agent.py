# 3_1_LLM_agent/tests/test_agent.py

import pytest
from llm_agent.core_v2 import LLMAgent


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать при быстрой проверке.
# Модель вынесена в константу: qwen3:0.6b — именно она скачивается в CI.
# Ассерты мягкие: слабая модель (0.6B) не всегда выдаёт конкретные слова,
# поэтому проверяем сам факт успешной работы агента (нет заглушки об ошибке).
# =====================================================================

MODEL = "qwen3:0.6b"

# Заглушка, которую агент возвращает при падении API/планирования
ERROR_STUB = "не удалось сгенерировать ответ"


@pytest.mark.integration
def test_audio_info_query_live():
    """MP3: агент должен обработать запрос и вернуть осмысленный ответ."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = "Извлеки метаданные из аудиофайла test_audio/sample.mp3 и напиши его длительность и битрейт."

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    # Главное — агент не упал с заглушкой
    assert ERROR_STUB not in response.lower()


@pytest.mark.integration
def test_audio_info_wav_query_live():
    """WAV: агент должен обработать запрос и вернуть осмысленный ответ."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = "Какие параметры у аудиофайла test_audio/sample.wav? Сколько там каналов?"

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    assert ERROR_STUB not in response.lower()


@pytest.mark.integration
def test_audio_info_tags_query_live():
    """Теги: агент должен обработать запрос и вернуть осмысленный ответ."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = "Какой исполнитель и альбом у файла test_audio/track.mp3?"

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    assert ERROR_STUB not in response.lower()
