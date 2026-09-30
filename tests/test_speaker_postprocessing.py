"""Behavioral checks for post-diarization speaker cleanup."""

from __future__ import annotations

import unittest
from copy import deepcopy

from local_transcriber.speaker_postprocessing import postprocess_speakers


def segment(start: float, end: float, text: str, speaker: str | None,
            *, status: str | None = None, candidates: list[str] | None = None) -> dict:
    return {
        "start": start, "end": end, "text": text, "speaker": speaker,
        "status": status or ("assigned" if speaker else "unassigned"),
        "candidate_speakers": candidates or [],
    }


A = "speaker_1"
B = "speaker_2"
C = "speaker_3"


class SpeakerPostprocessingTest(unittest.TestCase):
    def test_single_null_inside_a_sentence_is_restored_and_merged(self) -> None:
        groups = [segment(0, 0.6, " I think", A),
                  segment(0.65, 0.9, " that", None),
                  segment(0.95, 1.7, " works.", A)]
        original = deepcopy(groups)
        self.assertEqual(postprocess_speakers(groups), [
            segment(0, 1.7, " I think that works.", A),
        ])
        self.assertEqual(groups, original)

    def test_chain_of_nulls_is_restored_as_one_turn(self) -> None:
        groups = [segment(0, 0.4, " This", A),
                  segment(0.45, 0.6, " is", None),
                  segment(0.62, 0.8, " still", None),
                  segment(0.84, 1.4, " happening.", A)]
        self.assertEqual(postprocess_speakers(groups), [
            segment(0, 1.4, " This is still happening.", A),
        ])

    def test_null_on_speaker_boundary_is_not_guessed(self) -> None:
        groups = [segment(0, 0.4, " I think", A),
                  segment(0.45, 0.6, " so", None, candidates=[A]),
                  segment(0.65, 1.2, " but actually", B)]
        self.assertEqual(postprocess_speakers(groups), groups)

    def test_short_wrong_speaker_in_sentence_is_corrected(self) -> None:
        groups = [segment(0, 0.5, " I think", A),
                  segment(0.55, 0.8, " that", B),
                  segment(0.82, 1.3, " works.", A)]
        self.assertEqual(postprocess_speakers(groups), [
            segment(0, 1.3, " I think that works.", A),
        ])

    def test_real_short_interruption_is_preserved(self) -> None:
        for response in (" yeah", " right", " exactly", " no", " wow"):
            with self.subTest(response=response):
                groups = [segment(0, 0.5, " I think", A),
                          segment(0.52, 0.7, response, B),
                          segment(0.72, 1.2, " this works.", A)]
                self.assertEqual(postprocess_speakers(groups), groups)
        # A punctuation-delimited interjection is also a separate turn.
        groups = [segment(0, 0.5, " I think", A),
                  segment(0.52, 0.7, " Wait!", B),
                  segment(0.72, 1.2, " This works.", A)]
        self.assertEqual(postprocess_speakers(groups), groups)

    def test_russian_sentence_bridges_and_merges(self) -> None:
        null_groups = [segment(0, 0.4, " Я думаю", A),
                       segment(0.44, 0.55, " что", None),
                       segment(0.57, 0.76, " всё", None),
                       segment(0.8, 1.2, " получится.", A)]
        self.assertEqual(postprocess_speakers(null_groups), [
            segment(0, 1.2, " Я думаю что всё получится.", A),
        ])
        switch_groups = [segment(0, 0.4, " Я думаю", A),
                         segment(0.44, 0.55, " что", B),
                         segment(0.57, 1.2, " получится.", A)]
        self.assertEqual(postprocess_speakers(switch_groups), [
            segment(0, 1.2, " Я думаю что получится.", A),
        ])

    def test_russian_short_replies_are_not_reassigned(self) -> None:
        for response in (" да", " нет", " ага", " угу", " верно", " точно",
                         " именно", " ого", " хорошо", " окей", " неа",
                         " «да»", " Да…"):
            with self.subTest(response=response):
                for speaker in (B, None):
                    groups = [segment(0, 0.4, " Я думаю", A),
                              segment(0.43, 0.6, response, speaker),
                              segment(0.63, 1, " что получится.", A)]
                    self.assertEqual(postprocess_speakers(groups), groups)

    def test_russian_uncertain_boundary_and_separate_turns(self) -> None:
        boundary = [segment(0, 0.4, " Я думаю", A),
                    segment(0.44, 0.55, " что", None),
                    segment(0.6, 1.2, " получится.", B)]
        self.assertEqual(postprocess_speakers(boundary), boundary)
        sentences = [segment(0, 0.4, " Это всё.", A),
                     segment(0.44, 0.9, " Другая мысль.", A),
                     segment(0.95, 1.3, " Понятно.", B)]
        self.assertEqual(postprocess_speakers(sentences), sentences)

    def test_unchanged_sequence_and_sentence_boundaries(self) -> None:
        groups = [segment(0, 0.4, " Hello.", A),
                  segment(0.45, 0.8, " Another thought.", A),
                  segment(1.4, 1.9, " Different turn.", B)]
        self.assertEqual(postprocess_speakers(groups), groups)
        self.assertEqual(postprocess_speakers([]), [])

    def test_null_at_beginning_and_end_remains_null(self) -> None:
        groups = [segment(0, 0.3, " Unknown", None),
                  segment(0.35, 0.8, " starts here.", A),
                  segment(0.85, 1.2, " Unknown", None)]
        self.assertEqual(postprocess_speakers(groups), groups)

    def test_multiple_speaker_turns_keep_their_order(self) -> None:
        groups = [segment(0, 0.4, " First", A),
                  segment(0.45, 0.9, " continues.", A),
                  segment(1, 1.5, " Second.", B),
                  segment(1.55, 2, " Third.", C)]
        self.assertEqual(postprocess_speakers(groups), [
            segment(0, 0.9, " First continues.", A),
            *groups[2:],
        ])

    def test_uncertain_or_disconnected_spans_are_not_reassigned(self) -> None:
        cases = [
            [segment(0, 0.4, " I think", A),
             segment(0.45, 0.6, " that", None, status="ambiguous", candidates=[A, B]),
             segment(0.65, 1, " works.", A)],
            [segment(0, 0.4, " I think", A),
             segment(1, 1.2, " that", None),
             segment(1.25, 1.5, " works.", A)],
            [segment(0, 0.4, " I think", A),
             segment(0.45, 1.4, " this whole sentence", B),
             segment(1.45, 2, " works.", A)],
            [segment(0, 0.4, " I think", A),
             segment(0.45, 0.6, " that", None, candidates=[B]),
             segment(0.65, 1, " works.", A)],
        ]
        for groups in cases:
            with self.subTest(groups=groups):
                self.assertEqual(postprocess_speakers(groups), groups)


if __name__ == "__main__":
    unittest.main()
