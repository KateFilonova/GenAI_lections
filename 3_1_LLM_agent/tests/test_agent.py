# GenAI_lections/3_1_LLM_agent/tests/test_agent.py

import re
import pytest
from pathlib import Path
from llm_agent.tool_audioinfo import AudioInfoTool
from llm_agent.core_v2 import LLMAgent


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (реальная Ollama + реальный WAV)
# =====================================================================
# Файл: GenAI_lections/01_dialogue.wav
# Метаданные:
#   duration     ≈ 226.96 s
#   sample_rate  = 16000 Hz
#   channels     = 1 (моно)
#   bitrate      = 512000 bps
# =====================================================================
@pytest.mark.integration
def test_audio_info_real_wav():
    """Проверяет AudioInfoTool на реальном WAV-файле."""

    assert AUDIO_FILE.exists(), f"Нет файла: {AUDIO_FILE}"
    assert AUDIO_FILE.is_file(), f"Это не файл: {AUDIO_FILE}"

    tool = AudioInfoTool()
    response = tool.use(str(AUDIO_FILE))

    assert isinstance(response, str)
    assert "01_dialogue.wav" in response
    assert "WAV" in response

    assert "16000 Гц" in response
    assert "1" in response
    assert "512.0 кбит/с" in response
    assert "03:46" in response
MODEL = "qwen3:0.6b"

# Абсолютный путь к файлу — от корня репозитория
AUDIO_FILE = Path(__file__).parent.parent.parent / "01_dialogue.wav"

# Фразы, сигнализирующие об ошибке агента
ERROR_PHRASES = [
    "не удалось сгенерировать ответ",
    "не могу предоставить",
    "не найден",
    "not found",
    "error",
]

EXPECTED_DURATION = 226.96
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_BITRATE = 512000


def _extract_numbers(text: str) -> list[float]:
    """Вытаскивает все числа (целые и дробные) из строки."""
    return [float(n) for n in re.findall(r"\d+\.\d+|\d+", text)]


def _assert_no_error(response: str):
    """Проверяет, что агент не вернул ошибку."""
    lower = response.lower()
    for phrase in ERROR_PHRASES:
        assert phrase not in lower, f"Агент вернул ошибку: {response}"


@pytest.mark.integration
def test_audio_fixture_exists():
    """Убеждаемся, что файл на месте."""
    assert AUDIO_FILE.exists(), f"Нет файла {AUDIO_FILE}"


@pytest.mark.integration
def test_audio_info_query_live():
    """MP3/WAV: длительность и битрейт."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        f"Извлеки метаданные из аудиофайла {AUDIO_FILE} "
        "и напиши его длительность и битрейт."
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    _assert_no_error(response)

    assert any(w in response.lower() for w in [
        "длительн", "секунд", "битрейт", "kbps", "duration", "bitrate"
    ]), f"Ответ не про метаданные: {response}"

    assert re.search(r"\d", response), f"В ответе нет чисел: {response}"


@pytest.mark.integration
def test_audio_info_wav_channels_live():
    """WAV: каналы и частота дискретизации."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        f"Какие параметры у аудиофайла {AUDIO_FILE}? "
        "Сколько там каналов и какая частота дискретизации?"
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    _assert_no_error(response)

    assert any(w in response.lower() for w in [
        "канал", "моно", "частот", "гц", "hz", "sample", "channels"
    ]), f"Ответ не про каналы/частоту: {response}"


@pytest.mark.integration
def test_audio_info_full_info_live():
    """WAV: полные метаданные в одном ответе."""
    agent = LLMAgent(local=True, ollama_model=MODEL)
    query = (
        f"Извлеки полные метаданные из {AUDIO_FILE}: "
        "длительность, битрейт, количество каналов, частоту дискретизации."
    )
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0
    _assert_no_error(response)

    numbers = _extract_numbers(response)
    expected = [EXPECTED_DURATION, EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS, EXPECTED_BITRATE]
    found = []
    for exp in expected:
        tol = max(1.0, exp * 0.01)  # допуск 1%
        if any(abs(n - exp) <= tol for n in numbers):
            found.append(exp)

    assert len(found) >= 2, \
        f"Найдено меньше 2 ожидаемых значений. Ответ: {response}, числа: {numbers}"
