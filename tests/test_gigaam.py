from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from local_transcriber.gigaam import (
    MAX_CHUNK,
    MAX_CHUNK_SAMPLES,
    SAMPLE_RATE,
    Features,
    GigaAMEngine,
    _duration,
    _read_wav,
    _run_ffmpeg,
    _wav_frame_count,
    chunk_bounds,
    chunk_sample_bounds,
    find_silences,
)

PROFILE_CONFIG = {
    "preprocessor": {
        "sample_rate": SAMPLE_RATE,
        "features": 64,
        "n_fft": 320,
        "win_length": 320,
        "hop_length": 160,
        "center": False,
    }
}


def write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    channels: int = 1,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    pcm = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class GigaAMImportTest(unittest.TestCase):
    def test_import_does_not_load_model_runtime_dependencies(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import local_transcriber.gigaam; "
                    "assert not {'onnxruntime', 'sentencepiece', 'yaml'} "
                    "& sys.modules.keys()"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class GigaAMChunkingTest(unittest.TestCase):
    def assert_complete_sample_bounds(
        self, bounds: list[tuple[int, int]], total_samples: int
    ) -> None:
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], total_samples)
        for index, (start, end) in enumerate(bounds):
            self.assertLessEqual(end - start, MAX_CHUNK_SAMPLES)
            if total_samples:
                self.assertLess(start, end)
            if index:
                self.assertEqual(bounds[index - 1][1], start)

    def test_short_recording_stays_in_one_chunk(self) -> None:
        self.assertEqual(chunk_bounds(12.0, []), [(0.0, 12.0)])

    def test_uses_last_pause_before_model_limit(self) -> None:
        bounds = chunk_bounds(40.0, [5.0, 17.0, 23.0, 31.0])
        self.assertEqual(bounds[0], (0.0, 23.0))
        self.assertEqual(bounds[1], (23.0, 40.0))

    def test_falls_back_to_hard_limit_without_pause(self) -> None:
        bounds = chunk_bounds(50.0, [])
        self.assertEqual(bounds[0], (0.0, MAX_CHUNK))
        self.assertEqual(bounds[1], (MAX_CHUNK, MAX_CHUNK * 2))
        self.assertEqual(bounds[2], (MAX_CHUNK * 2, 50.0))

    def test_sample_boundaries_cover_limits_without_overlap_or_loss(self) -> None:
        totals = [
            MAX_CHUNK_SAMPLES,
            MAX_CHUNK_SAMPLES + 1,
            int(24.5 * SAMPLE_RATE),
            25 * SAMPLE_RATE,
            26 * SAMPLE_RATE,
            50 * SAMPLE_RATE + 7,
        ]
        for total_samples in totals:
            with self.subTest(total_samples=total_samples):
                bounds = chunk_sample_bounds(total_samples, [])
                self.assert_complete_sample_bounds(bounds, total_samples)
                expected_chunks = (
                    total_samples + MAX_CHUNK_SAMPLES - 1
                ) // MAX_CHUNK_SAMPLES
                self.assertEqual(len(bounds), expected_chunks)

    def test_sample_boundaries_prefer_pause_and_keep_short_tail(self) -> None:
        total_samples = 40 * SAMPLE_RATE + 1
        pause = 23 * SAMPLE_RATE
        bounds = chunk_sample_bounds(
            total_samples,
            [5 * SAMPLE_RATE, 17 * SAMPLE_RATE, pause, 31 * SAMPLE_RATE],
        )
        self.assertEqual(bounds, [(0, pause), (pause, total_samples)])
        self.assert_complete_sample_bounds(bounds, total_samples)


class AudioToolTest(unittest.TestCase):
    def test_ffmpeg_and_ffprobe_commands_are_explicit(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="1.25\n", stderr="")
        with patch(
            "local_transcriber.gigaam.subprocess.run", return_value=completed
        ) as run:
            _run_ffmpeg("-i", "source.wav", "-ac", "1", "-ar", "16000", "out.wav")
            self.assertEqual(
                run.call_args.args[0],
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    "source.wav",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "out.wav",
                ],
            )
            self.assertTrue(run.call_args.kwargs["check"])

        path = Path("converted.wav")
        with patch(
            "local_transcriber.gigaam.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(_duration(path), 1.25)
            self.assertEqual(run.call_args.args[0][0], "ffprobe")
            self.assertEqual(run.call_args.args[0][-1], str(path))
            self.assertTrue(run.call_args.kwargs["check"])

    def test_missing_and_failed_audio_tools_have_actionable_errors(self) -> None:
        with (
            patch(
                "local_transcriber.gigaam.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            self.assertRaisesRegex(ValueError, "ffmpeg not found"),
        ):
            _run_ffmpeg("-i", "source.wav", "out.wav")

        with (
            patch(
                "local_transcriber.gigaam.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            self.assertRaisesRegex(ValueError, "ffprobe not found"),
        ):
            _duration(Path("audio.wav"))

        failed = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr="decoder failed"
        )
        with (
            patch("local_transcriber.gigaam.subprocess.run", side_effect=failed),
            self.assertRaisesRegex(ValueError, "conversion failed: decoder failed"),
        ):
            _run_ffmpeg("-i", "source.wav", "out.wav")

        failed_probe = subprocess.CalledProcessError(
            1, ["ffprobe"], stderr="invalid media"
        )
        with (
            patch(
                "local_transcriber.gigaam.subprocess.run", side_effect=failed_probe
            ),
            self.assertRaisesRegex(ValueError, "ffprobe failed: invalid media"),
        ):
            _duration(Path("audio.wav"))

    def test_silencedetect_distinguishes_no_pauses_from_failure(self) -> None:
        no_pauses = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch(
            "local_transcriber.gigaam.subprocess.run", return_value=no_pauses
        ):
            self.assertEqual(find_silences(Path("audio.wav")), [])

        detected = subprocess.CompletedProcess(
            [],
            0,
            stdout="",
            stderr=(
                "[silencedetect] silence_start: 10\n"
                "[silencedetect] silence_end: 12 | silence_duration: 2\n"
            ),
        )
        with patch(
            "local_transcriber.gigaam.subprocess.run", return_value=detected
        ):
            self.assertEqual(find_silences(Path("audio.wav")), [11.0])

        failed = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr="filter unavailable"
        )
        with (
            patch("local_transcriber.gigaam.subprocess.run", side_effect=failed),
            self.assertRaisesRegex(
                ValueError, "silencedetect failed: filter unavailable"
            ),
        ):
            find_silences(Path("audio.wav"))


class WavInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reads_exact_mono_16khz_sample_range(self) -> None:
        path = self.root / "mono.wav"
        pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
        write_wav(path, pcm)

        self.assertEqual(_wav_frame_count(path), len(pcm))
        actual = _read_wav(path, 1, 4)
        np.testing.assert_allclose(actual, pcm[1:4].astype(np.float32) / 32768.0)

    def test_rejects_stereo_and_wrong_sample_rate_after_conversion(self) -> None:
        stereo = self.root / "stereo.wav"
        write_wav(stereo, np.zeros(8, dtype=np.int16), channels=2)
        wrong_rate = self.root / "wrong-rate.wav"
        write_wav(wrong_rate, np.zeros(8, dtype=np.int16), sample_rate=8_000)

        with self.assertRaisesRegex(ValueError, "16-bit mono"):
            _wav_frame_count(stereo)
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            _wav_frame_count(wrong_rate)


class FeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = Features(PROFILE_CONFIG)

    def test_fixed_profile_parameters_and_frame_lengths(self) -> None:
        self.assertEqual(self.features.sample_rate, 16_000)
        self.assertEqual(self.features.n_mels, 64)
        self.assertEqual(self.features.n_fft, 320)
        self.assertEqual(self.features.win_length, 320)
        self.assertEqual(self.features.hop_length, 160)
        self.assertFalse(self.features.center)
        for samples, expected in [(0, 0), (319, 0), (320, 1), (479, 1), (480, 2)]:
            with self.subTest(samples=samples):
                self.assertEqual(self.features.out_len(samples), expected)
                self.assertEqual(
                    self.features(np.zeros(samples, dtype=np.float32)).shape,
                    (1, 64, expected),
                )

    def test_silence_features_match_log_clip_contract(self) -> None:
        actual = self.features(np.zeros(320, dtype=np.float32))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, np.log(np.float32(1e-9)), rtol=1e-6)

    def test_synthetic_waveform_matches_pinned_preprocessing(self) -> None:
        wave = np.sin(
            2
            * np.pi
            * 440
            * np.arange(640, dtype=np.float32)
            / SAMPLE_RATE
        ).astype(np.float32)
        actual = self.features(wave)
        mel_indices = [0, 1, 5, 10, 20, 40, 63]

        self.assertEqual(actual.shape, (1, 64, 3))
        np.testing.assert_allclose(
            actual[0, mel_indices, 0],
            [
                -8.437281608581543,
                -7.193413257598877,
                -4.584892272949219,
                6.846816539764404,
                -7.336370944976807,
                -16.26276206970215,
                -20.7232666015625,
            ],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual[0, mel_indices, 1],
            [
                -7.653045177459717,
                -6.409177303314209,
                -4.473856449127197,
                6.846696376800537,
                -7.416224479675293,
                -17.264570236206055,
                -20.7232666015625,
            ],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(float(actual.sum()), -2239.729248046875, places=4)


class InferenceInputTest(unittest.TestCase):
    def build_engine(self) -> tuple[GigaAMEngine, Mock]:
        engine = object.__new__(GigaAMEngine)
        engine.features = Features(PROFILE_CONFIG)
        encoder = Mock()
        engine.encoder = encoder
        encoder.get_outputs.return_value = [
            SimpleNamespace(name="encoded"),
            SimpleNamespace(name="encoded_len"),
        ]
        encoder.get_inputs.return_value = [
            SimpleNamespace(name="audio_signal"),
            SimpleNamespace(name="length"),
        ]
        encoder.run.return_value = [
            np.zeros((1, 768, 1), dtype=np.float32),
            np.array([0], dtype=np.int32),
        ]
        engine.decoder = Mock()
        engine.decoder.get_outputs.return_value = []
        engine.decoder.get_inputs.return_value = []
        engine.joint = Mock()
        engine.joint.get_outputs.return_value = []
        engine.joint.get_inputs.return_value = []
        engine.blank_id = 0
        engine.pred_layers = 1
        engine.pred_hidden = 1
        engine.tokenizer = Mock()
        engine.tokenizer.decode.return_value = ""
        return engine, encoder

    def test_empty_and_short_pcm_do_not_call_onnx(self) -> None:
        engine, encoder = self.build_engine()
        for samples in (0, 319):
            with self.subTest(samples=samples):
                self.assertEqual(
                    engine.transcribe_wave(np.zeros(samples, dtype=np.float32)), ""
                )
        encoder.run.assert_not_called()

    def test_exact_window_and_silence_use_non_negative_length(self) -> None:
        engine, encoder = self.build_engine()
        self.assertEqual(engine.transcribe_wave(np.zeros(320, dtype=np.float32)), "")
        feed = encoder.run.call_args.args[1]
        np.testing.assert_array_equal(feed["length"], np.array([1], dtype=np.int64))
        self.assertEqual(feed["audio_signal"].shape, (1, 64, 1))


class BatchTranscriptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_fake_batch(
        self,
        source: Path,
        total_samples: int,
        silences: list[float],
        *,
        reported_duration: float | None = None,
    ) -> tuple[list, Mock, Mock, Mock]:
        engine = object.__new__(GigaAMEngine)
        engine.transcribe_wave = Mock(
            side_effect=lambda audio: f"samples-{len(audio)}"
        )

        def convert(*args: str) -> None:
            self.assertEqual(
                args,
                (
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    args[-1],
                ),
            )
            write_wav(Path(args[-1]), np.zeros(total_samples, dtype=np.int16))

        converter = Mock(side_effect=convert)
        duration = Mock(
            return_value=(
                total_samples / SAMPLE_RATE
                if reported_duration is None
                else reported_duration
            )
        )
        silence_detector = Mock(return_value=silences)
        with (
            patch("local_transcriber.gigaam._run_ffmpeg", converter),
            patch("local_transcriber.gigaam._duration", duration),
            patch("local_transcriber.gigaam.find_silences", silence_detector),
        ):
            segments = engine.transcribe(source)
        return segments, converter, duration, silence_detector

    def test_arbitrary_source_is_converted_and_24s_plus_one_sample_is_split(self):
        sources = [
            ("mono-8k.wav", 1, 8_000),
            ("stereo-48k.wav", 2, 48_000),
        ]
        for name, channels, sample_rate in sources:
            with self.subTest(name=name):
                source = self.root / name
                write_wav(
                    source,
                    np.zeros(channels * 8, dtype=np.int16),
                    channels=channels,
                    sample_rate=sample_rate,
                )
                segments, converter, duration, silence_detector = self.run_fake_batch(
                    source,
                    MAX_CHUNK_SAMPLES + 1,
                    [],
                    reported_duration=1.0,
                )
                self.assertEqual(
                    [(item.start, item.end, item.text) for item in segments],
                    [
                        (0.0, 24.0, f"samples-{MAX_CHUNK_SAMPLES}"),
                        (24.0, 24.0 + 1 / SAMPLE_RATE, "samples-1"),
                    ],
                )
                converter.assert_called_once()
                duration.assert_called_once()
                silence_detector.assert_called_once()

    def test_full_result_duration_uses_wav_frames_even_without_speech(self) -> None:
        cases = [
            (0, [], []),
            (159, [""], []),
            (SAMPLE_RATE, [""], []),
            (MAX_CHUNK_SAMPLES + 1, ["speech", ""], [(0.0, 24.0)]),
            (MAX_CHUNK_SAMPLES + 1, ["", ""], []),
        ]
        for total_samples, texts, expected_ranges in cases:
            with self.subTest(total_samples=total_samples, texts=texts):
                engine = object.__new__(GigaAMEngine)
                engine.transcribe_wave = Mock(side_effect=texts)

                def convert(*args: str, frames: int = total_samples) -> None:
                    write_wav(Path(args[-1]), np.zeros(frames, dtype=np.int16))

                with (
                    patch("local_transcriber.gigaam._run_ffmpeg", side_effect=convert),
                    patch("local_transcriber.gigaam._duration", return_value=999.0),
                    patch("local_transcriber.gigaam.find_silences", return_value=[]),
                ):
                    result = engine.transcribe_result(self.root / "synthetic.wav")
                self.assertEqual(result.duration_seconds, total_samples / SAMPLE_RATE)
                self.assertEqual(result.language, "ru")
                self.assertEqual(
                    [(s["start"], s["end"]) for s in result.raw_segments],
                    expected_ranges,
                )
                self.assertEqual(engine.transcribe_wave.call_count, len(texts))

    def test_pause_split_reads_contiguous_ranges_with_short_tail(self) -> None:
        source = self.root / "long.wav"
        write_wav(source, np.zeros(8, dtype=np.int16))
        total_samples = 40 * SAMPLE_RATE + 1
        segments, _, _, _ = self.run_fake_batch(source, total_samples, [23.0])
        self.assertEqual(
            [(item.start, item.end, item.text) for item in segments],
            [
                (0.0, 23.0, f"samples-{23 * SAMPLE_RATE}"),
                (
                    23.0,
                    total_samples / SAMPLE_RATE,
                    f"samples-{total_samples - 23 * SAMPLE_RATE}",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
