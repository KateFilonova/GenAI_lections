from pathlib import Path
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE


class AudioInfoTool:
    """Инструмент для извлечения метаданных аудиофайла (MP3, WAV)."""

    name = "audio_info"
    description = (
        "Извлекает метаданные аудиофайла (MP3, WAV): длительность, битрейт, "
        "частоту дискретизации, количество каналов, теги (исполнитель, альбом, название). "
        "Принимает путь к файлу."
    )

    # Поддерживаемые расширения
    SUPPORTED_FORMATS = {".mp3", ".wav"}

    def use(self, file_path: str) -> str:
        """
        Извлекает метаданные аудиофайла по указанному пути.

        Args:
            file_path: Путь к аудиофайлу (MP3 или WAV).

        Returns:
            Отформатированная строка с метаданными или сообщение об ошибке.
        """
        try:
            path = Path(file_path)

            # --- 1. Проверка существования файла ---
            if not path.exists():
                return f"Ошибка: файл '{file_path}' не найден."

            if not path.is_file():
                return f"Ошибка: '{file_path}' не является файлом."

            # --- 2. Проверка расширения ---
            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                return (
                    f"Ошибка: формат '{ext}' не поддерживается. "
                    f"Доступные форматы: {', '.join(sorted(self.SUPPORTED_FORMATS))}."
                )

            print(f"> Извлекаю метаданные аудиофайла: '{path.name}'")

            # --- 3. Чтение метаданных через mutagen ---
            audio = MutagenFile(path)

            if audio is None:
                return f"Ошибка: не удалось прочитать файл '{file_path}'. Возможно, он повреждён."

            if audio.info is None:
                return f"Ошибка: не удалось получить информацию о аудиопотоке в '{file_path}'."

            info = audio.info

            # --- 4. Общие параметры ---
            duration_sec = getattr(info, "length", 0.0)
            duration_str = self._format_duration(duration_sec)

            sample_rate = getattr(info, "sample_rate", None)
            channels = getattr(info, "channels", None)
            bitrate = getattr(info, "bitrate", None)

            # Для WAV mutagen использует битрейт как bits_per_sample * sample_rate * channels,
            # но у WAVE-объекта есть отдельное поле bits_per_sample
            bits_per_sample = getattr(info, "bits_per_sample", None)

            # --- 5. Теги (зависят от формата) ---
            tags = self._extract_tags(audio, ext)

            # --- 6. Формируем отчёт ---
            lines = [
                f"**Метаданные аудиофайла:** {path.name}",
                f"**Формат:** {ext.lstrip('.').upper()}",
                f"**Длительность:** {duration_str} ({duration_sec:.2f} сек.)",
                f"**Битрейт:** {self._format_bitrate(bitrate)}",
                f"**Частота дискретизации:** {sample_rate} Гц" if sample_rate else "**Частота дискретизации:** н/д",
                f"**Количество каналов:** {channels}" if channels else "**Количество каналов:** н/д",
            ]

            if bits_per_sample:
                lines.append(f"**Разрядность:** {bits_per_sample} бит")

            if tags:
                lines.append("")
                lines.append("**Теги:**")
                for key, value in tags.items():
                    lines.append(f"  - {key}: {value}")
            else:
                lines.append("")
                lines.append("**Теги:** отсутствуют")

            # Размер файла
            size_bytes = path.stat().st_size
            lines.append("")
            lines.append(f"**Размер файла:** {self._format_size(size_bytes)}")

            result = "\n".join(lines)
            print(f"> Метаданные успешно извлечены: {path.name}")
            return result

        except Exception as e:
            print(f"> Ошибка при извлечении метаданных: {e}")
            return f"Произошла ошибка при чтении файла '{file_path}': {e}"

    # ---------- Вспомогательные методы ----------

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Преобразует секунды в формат 'мм:сс' или 'чч:мм:сс'."""
        if not seconds or seconds < 0:
            return "н/д"
        total = int(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_bitrate(bitrate) -> str:
        """Преобразует битрейт (бит/с) в читаемый вид."""
        if not bitrate:
            return "н/д"
        return f"{bitrate / 1000:.1f} кбит/с"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Преобразует размер файла в читаемый вид."""
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} ТБ"

    @staticmethod
    def _extract_tags(audio, ext: str) -> dict:
        """
        Извлекает теги из аудиофайла.
        Для MP3 используется ID3, для WAV — RIFF INFO.
        """
        tags = {}

        # ID3 (MP3)
        if ext == ".mp3" and hasattr(audio, "tags") and audio.tags:
            tag_map = {
                "TIT2": "Название",
                "TPE1": "Исполнитель",
                "TALB": "Альбом",
                "TDRC": "Год",
                "TCON": "Жанр",
                "TRCK": "Трек",
            }
            for frame_id, label in tag_map.items():
                frame = audio.tags.get(frame_id)
                if frame:
                    tags[label] = str(frame)

        # RIFF INFO (WAV)
        elif ext == ".wav" and hasattr(audio, "tags") and audio.tags:
            info_map = {
                "INAM": "Название",
                "IART": "Исполнитель",
                "IPRD": "Альбом",
                "ICRD": "Дата",
                "IGNR": "Жанр",
                "ICMT": "Комментарий",
            }
            for tag_id, label in info_map.items():
                value = audio.tags.get(tag_id)
                if value:
                    tags[label] = str(value[0]) if isinstance(value, list) else str(value)

        return tags
