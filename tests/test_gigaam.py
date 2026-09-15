from __future__ import annotations

import unittest

from local_transcriber.gigaam import MAX_CHUNK, chunk_bounds


class GigaAMChunkingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
