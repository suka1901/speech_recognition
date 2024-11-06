#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import speech_recognition as sr


class TestSpecialFeatures(unittest.TestCase):
    def setUp(self):
        self.AUDIO_FILE_EN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "english.wav")
        self.AUDIO_FILE_EN_MP3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "english.mp3")
        self.addTypeEqualityFunc(str, self.assertSameWords)

    # @unittest.skipIf(sys.platform.startswith("win"), "skip on Windows")
    def test_sphinx_keywords(self):
        r = sr.Recognizer()
        with sr.AudioFile(self.AUDIO_FILE_EN) as source: audio = r.record(source)
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("one", 0.01), ("two", 0.02), ("three", 0.03)]), "three two one")
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("wan", 0.01), ("too", 0.02), ("tree", 0.03)]), "tree too wan")
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("un", 0.01), ("to", 0.02), ("tee", 0.03)]), "tee to un")

    def assertSameWords(self, tested, reference, msg=None):
        set_tested = set(tested.split())
        set_reference = set(reference.split())
        if set_tested != set_reference:
            raise self.failureException(msg if msg is not None else "%r doesn't consist of the same words as %r" % (tested, reference))

       # added test for MP3
    def test_incompatible_audio_file_error(self):
        r = sr.Recognizer()

        with self.assertRaises(ValueError) as context:
            with sr.AudioFile(self.AUDIO_FILE_EN_MP3) as source:r.record(source)

        self.assertEqual(
            str(context.exception),
            "Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format"
        )

if __name__ == "__main__":
    unittest.main()
