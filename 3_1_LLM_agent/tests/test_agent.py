import re
from pathlib import Path

import pytest

from llm_agent.core_v2 import LLMAgent
from llm_agent.tool_audioinfo import AudioInfoTool


# =====================================================================
# НАСТРОЙКИ
# =====================================================================

MODEL = "qwen3:0.6b"

# test_agent.py находится:
# GenAI_lections/3_1_LLM_agent/tests/test_agent.py
#
# Поэтому:
# parent        -> tests
# parent.parent -> 3_1_LLM_agent
# parent.parent.parent -> корень GenAI_lections
AUDIO_FILE = Path(__file__).parent.parent.parent / "01_dialogue.wav"


# Метаданные реального WAV-файла
EXPECTED_DURATION = 226.96
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_BITRATE = 512000


# Фразы, сигнализирующие об ошибке агента
ERROR_PHRASES = [
    "не удалось сгенерировать ответ",
    "не могу предоставить",
    "не найден",
    "not found",
    "error",
]


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def _extract_numbers(text: str) -> list[float]:
    """Вытаскивает все числа (целые и дробные) из строки."""
    return [float(n) for n in re.findall(r"\d+\.\d+|\d+", text)]


def _assert_no_error(response: str):
    """Проверяет, что агент не вернул ошибку."""
    lower = response.lower()

    for phrase in ERROR_PHRASES:
        assert phrase not in lower, f"Агент вернул ошибку: {response}"


# =====================================================================
# 1. ПРОВЕРКА НАЛИЧИЯ РЕАЛЬНОГО АУДИОФАЙЛА
# =====================================================================

@pytest.mark.integration
def test_audio_fixture_exists():
    """Проверяет, что реальный WAV-файл находится в репозитории."""

    print(f"\nИщем аудиофайл: {AUDIO_FILE}")
    print(f"Текущая директория: {Path.cwd()}")

    assert AUDIO_FILE.exists(), f"Нет файла: {AUDIO_FILE}"
    assert AUDIO_FILE.is_file(), f"Это не файл: {AUDIO_FILE}"


# =====================================================================
# 2. НЕПОСРЕДСТВЕННЫЙ ТЕСТ AudioInfoTool НА РЕАЛЬНОМ WAV
# =====================================================================

@pytest.mark.integration
def test_audio_info_real_wav():
    """Проверяет AudioInfoTool на реальном WAV-файле."""

    assert AUDIO_FILE.exists(), f"Нет файла: {AUDIO_FILE}"
    assert AUDIO_FILE.is_file(), f"Это не файл: {AUDIO_FILE}"

    tool = AudioInfoTool()

    response = tool.use(str(AUDIO_FILE))

    assert isinstance(response, str)
    assert len(response) > 0

    # Проверяем, что обработан именно нужный файл
    assert "01_dialogue.wav" in response

    # Проверяем формат
    assert "WAV" in response

    # Проверяем реальные параметры файла
    assert "16000 Гц" in response
    assert "**Количество каналов:** 1" in response
    assert "512.0 кбит/с" in response

    # 226.96 секунд отображаются как 03:46
    assert "03:46" in response


# =====================================================================
# 3. ИНТЕГРАЦИОННЫЙ ТЕСТ:
#    LLMAgent + Ollama + AudioInfoTool + реальный WAV
# =====================================================================

@pytest.mark.integration
def test_audio_info_query_live():
    """Проверяет извлечение длительности и битрейта через агента."""

    agent = LLMAgent(
        local=True,
        ollama_model=MODEL
    )

    query = (
        "Используй инструмент audio_info. "
        f"Извлеки метаданные именно из файла: {AUDIO_FILE}. "
        "Напиши его длительность и битрейт."
    )

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0

    _assert_no_error(response)

    assert any(
        word in response.lower()
        for word in [
            "длительн",
            "секунд",
            "битрейт",
            "kbps",
            "duration",
            "bitrate",
        ]
    ), f"Ответ не про метаданные: {response}"

    assert re.search(r"\d", response), (
        f"В ответе нет чисел: {response}"
    )


# =====================================================================
# 4. ИНТЕГРАЦИОННЫЙ ТЕСТ:
#    КАНАЛЫ И ЧАСТОТА ДИСКРЕТИЗАЦИИ
# =====================================================================

@pytest.mark.integration
def test_audio_info_wav_channels_live():
    """Проверяет каналы и частоту дискретизации через агента."""

    agent = LLMAgent(
        local=True,
        ollama_model=MODEL
    )

    query = (
        "Используй инструмент audio_info. "
        f"Обработай именно этот WAV-файл: {AUDIO_FILE}. "
        "Укажи количество каналов и частоту дискретизации."
    )

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0

    _assert_no_error(response)

    assert any(
        word in response.lower()
        for word in [
            "канал",
            "моно",
            "частот",
            "гц",
            "hz",
            "sample",
            "channels",
        ]
    ), f"Ответ не про каналы/частоту: {response}"


# =====================================================================
# 5. ИНТЕГРАЦИОННЫЙ ТЕСТ:
#    ПОЛНЫЕ МЕТАДАННЫЕ
# =====================================================================

@pytest.mark.integration
def test_audio_info_full_info_live():
    """Проверяет получение основных метаданных через агента."""

    agent = LLMAgent(
        local=True,
        ollama_model=MODEL
    )

    query = (
        "Используй инструмент audio_info. "
        f"Извлеки полные метаданные из файла: {AUDIO_FILE}. "
        "Укажи длительность, битрейт, количество каналов "
        "и частоту дискретизации."
    )

    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0

    _assert_no_error(response)

    numbers = _extract_numbers(response)

    expected = [
        EXPECTED_DURATION,
        EXPECTED_SAMPLE_RATE,
        EXPECTED_CHANNELS,
        EXPECTED_BITRATE,
    ]

    found = []

    for exp in expected:
        tolerance = max(1.0, exp * 0.01)

        if any(
            abs(number - exp) <= tolerance
            for number in numbers
        ):
            found.append(exp)

    assert len(found) >= 2, (
        f"Найдено меньше двух ожидаемых значений.\n"
        f"Ответ: {response}\n"
        f"Числа: {numbers}"
    )
