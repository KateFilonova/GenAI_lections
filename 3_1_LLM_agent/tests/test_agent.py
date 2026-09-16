# 3_1_LLM_agent/tests/test_agent.py

import re
import pytest
from pathlib import Path

from llm_agent.core_v2 import LLMAgent


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (реальная Ollama + реальный WAV-файл)
# =====================================================================
# Файл 01_dialogue.wav лежит в корне репозитория.
# Путь считается от 3_1_LLM_agent/, где запускается pytest:
#   Path(__file__)              = 3_1_LLM_agent/tests/test_agent.py
#   .parent                     = 3_1_LLM_agent/tests/
#   .parent.parent              = 3_1_LLM_agent/
#   .parent.parent.parent       = корень репо  ← здесь 01_dialogue.wav
#
# Метаданные файла:
#   duration ≈ 226.96 s
#   sample_rate = 16000 Hz
#   channels = 1 (моно)
#   bitrate = 512000 bps
# =====================================================================

MODEL = "qwen3:0.6b"
ERROR_STUB = "не удалось сгенерировать ответ"

# Путь к файлу в корне репозитория
AUDIO_FILE = Path(__file__).parent.parent.parent / "01_dialogue.wav"

EXPECTED_DURATION = 226.96
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_BITRATE = 512000


def _extract_numbers(text: str) -> list[float]:
    """Вытаскивает все числа (целые и дробные) из строки."""
    return [float(n) for n in re.findall(r"\d+\.\d+|\d+", text)]


@pytest.mark.integration
def test_audio_fixture_exists():
    """Убеждаемся, что файл на месте перед интеграционными тестами."""
    assert AUDIO_FILE.exists(), f"Нет файла {AUDIO_FILE}"


@pytest.mark.integration
def test_audio_info_query_live():
    """Агент должен вернуть осмысленный ответ про реальный WAV."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        "Извлеки метаданные из аудиофайла 01_dialogue.wav "
        "и напиши его длительность и битрейт."
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    assert ERROR_STUB not in response.lower()

    assert any(w in response.lower() for w in [
        "длительн", "секунд", "битрейт", "kbps", "duration", "bitrate"
    ]), f"Ответ не про метаданные: {response}"

    assert re.search(r"\d", response), f"В ответе нет чисел: {response}"


@pytest.mark.integration
def test_audio_info_wav_channels_live():
    """Агент должен вернуть осмысленный ответ про каналы WAV."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        "Какие параметры у аудиофайла 01_dialogue.wav? "
        "Сколько там каналов и какая частота дискретизации?"
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    assert ERROR_STUB not in response.lower()

    assert any(w in response.lower() for w in [
        "канал", "моно", "частот", "гц", "hz", "sample", "channels"
    ]), f"Ответ не про каналы/частоту: {response}"


@pytest.mark.integration
def test_audio_info_full_info_live():
    """Агент должен вернуть метаданные в одном ответе."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        "Извлеки полные метаданные из 01_dialogue.wav: "
        "длительность, битрейт, количество каналов, частоту дискретизации."
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    assert ERROR_STUB not in response.lower()

    numbers = _extract_numbers(response)
    expected = [EXPECTED_DURATION, EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS, EXPECTED_BITRATE]
    found = []
    for exp in expected:
        tol = max(1.0, exp * 0.01)
        if any(abs(n - exp) <= tol for n in numbers):
            found.append(exp)

    assert len(found) >= 2, \
        f"Найдено меньше 2 ожидаемых значений. Ответ: {response}, числа: {numbers}"
