import requests
import json
import re
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_pdfinfo import PDFInfoTool
from .tool_audioinfo import AudioInfoTool


class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(
        self,
        model: str = "tngtech/deepseek-r1t2-chimera",
        local: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3:0.6b"
    ):
        """
        Инициализирует агента.

        Args:
            model (str): Название модели для OpenRouter.
            local (bool): Если True, использует локальный Ollama вместо OpenRouter.
            ollama_base_url (str): Базовый URL для Ollama API.
            ollama_model (str): Название модели в Ollama.
        """
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        if not self.local:
            self.api_key = config('OPENROUTER_API_KEY')
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model

        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "pdf_info": PDFInfoTool(),
            "audio_info": AudioInfoTool(),
        }

        self.conversation_history = []

    def _make_api_request(
        self,
        payload: Dict,
        headers: Optional[Dict] = None
    ) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        Поддерживает как OpenRouter, так и Ollama.

        Args:
            payload (Dict): Тело запроса.
            headers (Dict, optional): Заголовки запроса.

        Returns:
            Dict: Ответ от API.
        """

        if headers is None:
            headers = {}

        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=headers
            )

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")

    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Работает как с OpenRouter, так и с Ollama.
        """

        # Системный промпт объясняет LLM,
        # какие инструменты доступны и как ими пользоваться.
        system_prompt = f"""
        You are a helpful AI planning assistant.
        Analyze the user's request and decide if you need to use any tools.

        Available tools:

        - **calculator**:
          For any math-related questions (numbers, calculations).
          Use it with the full expression.

        - **web_search**:
          For finding any information about the real world
          (current events, facts, definitions).
          Use it with the user's question or a clear search query.
          USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.

        - **pdf_info**:
          For extracting information from PDF files
          (metadata, page count, text content).
          Use it with a local file path or a URL to a PDF file.

        - **audio_info**:
          For extracting metadata from audio files (MP3, WAV):
          duration, bitrate, sample rate, channels,
          tags (artist, album, title).
          Use it with a local file path to an audio file.

        IMPORTANT RULE FOR audio_info:

        If the user asks about an MP3 or WAV file,
        the "input" field for audio_info should contain
        ONLY the file path.

        Do not include explanations, questions,
        or other text in the "input" field.

        Your response MUST be ONLY a JSON object
        of the following format.

        If one or more tools are needed, return:

        {{
            "plan": [
                {{"action": "tool_name", "input": "some text to pass into tool"}},
                ...
            ]
        }}

        One action corresponds to one tool call.

        If no tool is needed, return:

        {{"plan": []}}
        """

        # Формируем запрос к API
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        try:
            # Для Ollama отключаем потоковую передачу,
            # чтобы получить один готовый ответ.
            if self.local:
                payload["stream"] = False

            response_data = self._make_api_request(payload)

            # Извлекаем текстовый ответ от модели.
            llm_text = response_data["choices"][0]["message"]["content"]

            # Ищем JSON внутри блока Markdown-кода.
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```',
                llm_text,
                re.DOTALL
            )

            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = llm_text

            print(
                f"> Ответ LLM для плана (очищенный): "
                f"{cleaned_json_text}"
            )

            # Преобразуем JSON в Python-словарь.
            action_plan = json.loads(cleaned_json_text)

            plan = action_plan.get("plan", [])

            return plan

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(
                f"Произошла ошибка при создании плана: {e}"
            )

            # Пробуем найти JSON внутри обычного текста.
            try:
                json_match = re.search(
                    r'\{.*"plan".*\}',
                    llm_text,
                    re.DOTALL
                )

                if json_match:
                    action_plan = json.loads(
                        json_match.group()
                    )

                    return action_plan.get("plan", [])

            except Exception:
                pass

            return []

    @staticmethod
    def _extract_audio_path(tool_input: str) -> str:
        """
        Извлекает путь к MP3 или WAV из строки.

        Например, если LLM вернула:

        "Нужно обработать файл /home/user/music.wav"

        метод вернет:

        "/home/user/music.wav"
        """

        if not isinstance(tool_input, str):
            return tool_input

        text = tool_input.strip()

        # Ищем Linux или Windows путь,
        # заканчивающийся на .mp3 или .wav.
        match = re.search(
            r'(?P<path>(?:[A-Za-z]:[\\/]|/)?[^\n"\']*?\.(?:mp3|wav))',
            text,
            re.IGNORECASE
        )

        if match:
            audio_path = match.group("path").strip()

            # Удаляем возможные знаки препинания
            # после расширения файла.
            audio_path = audio_path.rstrip(
                ".,;:!?)]}"
            )

            return audio_path

        # Если путь не найден,
        # возвращаем исходное значение.
        return text

    @staticmethod
    def _extract_audio_path_from_query(query: str) -> Optional[str]:
        """
        Извлекает путь к MP3 или WAV непосредственно
        из исходного запроса пользователя.

        Это более надежный источник пути, чем ответ
        небольшой LLM, которая может изменить абсолютный путь.
        """

        if not isinstance(query, str):
            return None

        # Сначала ищем абсолютный Linux-путь.
        linux_match = re.search(
            r'(/[^\n"\']+\.(?:mp3|wav))',
            query,
            re.IGNORECASE
        )

        if linux_match:
            return linux_match.group(1).strip().rstrip(
                ".,;:!?)]}"
            )

        # Затем ищем Windows-путь.
        windows_match = re.search(
            r'([A-Za-z]:[\\/][^\n"\']+\.(?:mp3|wav))',
            query,
            re.IGNORECASE
        )

        if windows_match:
            return windows_match.group(1).strip().rstrip(
                ".,;:!?)]}"
            )

        # Если абсолютный путь не найден,
        # ищем просто имя аудиофайла.
        filename_match = re.search(
            r'([^\s"\'/:]+\.(?:mp3|wav))',
            query,
            re.IGNORECASE
        )

        if filename_match:
            return filename_match.group(1).strip().rstrip(
                ".,;:!?)]}"
            )

        return None

    def _generate_final_response(
        self,
        user_query: str
    ) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
        """

        prompt = f"""
        Based on the following conversation log,
        provide a direct and helpful answer to the user's
        original question.

        Be concise and use the information from the tool
        results to support your answer.

        Original User Question:
        {user_query}

        Conversation Log:
        {chr(10).join(
            [msg['content'] for msg in self.conversation_history]
        )}
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        if self.local:
            payload["stream"] = False

        try:
            response_data = self._make_api_request(
                payload
            )

            final_text = response_data[
                "choices"
            ][0]["message"]["content"]

            return final_text

        except Exception as e:
            return (
                "Ошибка при генерации финального ответа. "
                f"Детали: {e}"
            )

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        """

        print(
            f"Агент анализирует ваш запрос... "
            f"(Режим: "
            f"{'локальный Ollama' if self.local else 'OpenRouter'})"
        )

        # ---------------------------------------------------------
        # Шаг 1. Планирование
        # ---------------------------------------------------------

        plan = self._ask_llm_for_plan(query)

        if not plan:
            print(
                "Инструменты не требуются. "
                "Генерирую ответ напрямую."
            )

            # Генерируем прямой ответ через LLM.
            direct_prompt = (
                "Ответьте на следующий вопрос "
                f"кратко и информативно: {query}"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": direct_prompt
                    }
                ]
            }

            if self.local:
                payload["stream"] = False

            try:
                response_data = self._make_api_request(
                    payload
                )

                return response_data[
                    "choices"
                ][0]["message"]["content"]

            except Exception:
                return (
                    "Извините, не удалось "
                    "сгенерировать ответ."
                )

        # ---------------------------------------------------------
        # Шаг 2. Исполнение плана
        # ---------------------------------------------------------

        print(f"План действий: {plan}")

        for step in plan:
            tool_name = step.get("action")
            tool_input = step.get("input")

            # -----------------------------------------------------
            # ИСПРАВЛЕНИЕ ДЛЯ AUDIO_INFO
            # -----------------------------------------------------
            #
            # Маленькая LLM может:
            #
            # 1. правильно определить инструмент;
            # 2. но неправильно сформировать путь к файлу.
            #
            # Например, настоящий путь:
            #
            # /home/runner/work/GenAI_lections/
            # GenAI_lections/01_dialogue.wav
            #
            # LLM может ошибочно написать:
            #
            # /home/runner/work/GenAI_lections/
            # 01_dialogue.wav
            #
            # Поэтому для audio_info мы сначала берем
            # путь непосредственно из исходного запроса пользователя.
            # -----------------------------------------------------

            if tool_name == "audio_info":

                # Пытаемся найти настоящий путь
                # в исходном запросе пользователя.
                query_audio_path = (
                    self._extract_audio_path_from_query(
                        query
                    )
                )

                if query_audio_path:
                    tool_input = query_audio_path

                    print(
                        "> Используется путь к аудиофайлу "
                        "из запроса пользователя: "
                        f"{tool_input}"
                    )

                elif isinstance(tool_input, str):
                    # Если путь в исходном запросе
                    # найти не удалось, используем
                    # путь, который передала LLM.
                    original_input = tool_input

                    tool_input = (
                        self._extract_audio_path(
                            tool_input
                        )
                    )

                    print(
                        "> Извлечен путь к аудиофайлу "
                        "из ответа LLM: "
                        f"{tool_input}"
                    )

                    if original_input != tool_input:
                        print(
                            "> LLM передала дополнительный "
                            "текст. Используется только путь "
                            "к аудиофайлу."
                        )

            # -----------------------------------------------------
            # Проверяем, существует ли инструмент
            # -----------------------------------------------------

            if tool_name in self.tools:

                print(
                    f"Выполняется инструмент: "
                    f"'{tool_name}'"
                )

                result = self.tools[
                    tool_name
                ].use(tool_input)

                print(
                    f"Результат: {result}..."
                )

                # Добавляем результат работы инструмента
                # в историю разговора.
                self.conversation_history.append({
                    "role": "system",
                    "content": (
                        f"Tool {tool_name} result: "
                        f"{result}"
                    )
                })

            else:

                error_msg = (
                    f"Ошибка: инструмент с именем "
                    f"'{tool_name}' не найден."
                )

                print(error_msg)

                self.conversation_history.append({
                    "role": "system",
                    "content": error_msg
                })

        # ---------------------------------------------------------
        # Шаг 3. Генерация финального ответа
        # ---------------------------------------------------------

        print(
            "Составляю финальный ответ..."
        )

        final_response = self._generate_final_response(
            query
        )

        return final_response

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.

        Returns:
            bool: True если соединение успешно, иначе False.
        """

        if not self.local:
            return False

        try:
            # Проверяем доступность Ollama API.
            test_url = (
                f"{self.ollama_base_url}/v1/models"
            )

            response = requests.get(test_url)

            return response.status_code == 200

        except Exception:
            return False
