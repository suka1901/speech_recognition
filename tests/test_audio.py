#!/usr/bin/env python3
import audioop
import unittest
from os import path
from unittest import mock

from numpy.ma.testutils import assert_equal

import speech_recognition as sr


def assert_similar(bytes_1, bytes_2):
    for i, (byte_1, byte_2) in enumerate(zip(bytes_1, bytes_2)):
        if abs(byte_1 - byte_2) > 2:
            raise AssertionError("{} is really different from {} at index {}".format(bytes_1, bytes_2, i))


class TestAudioFile(unittest.TestCase):

    def test_get_segment(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-32-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertEqual(audio.get_raw_data(), audio.get_segment().get_raw_data())
        self.assertEqual(audio.get_raw_data()[8:], audio.get_segment(0.022675738 * 2).get_raw_data())
        self.assertEqual(audio.get_raw_data()[:16], audio.get_segment(None, 0.022675738 * 4).get_raw_data())
        self.assertEqual(audio.get_raw_data()[8:16], audio.get_segment(0.022675738 * 2, 0.022675738 * 4).get_raw_data())

    def test_wav_mono_8_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-8-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 1)
        assert_similar(audio.get_raw_data()[:32], b"\x00\xff\x00\xff\x00\xff\xff\x00\xff\x00\xff\x00\xff\x00\x00\xff\x00\x00\xff\x00\xff\x00\xff\x00\xff\x00\xff\x00\xff\x00\xff\xff")

    def test_wav_mono_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-16-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x01\x00\xfe\xff\x01\x00\xfe\xff\x04\x00\xfc\xff\x04\x00\xfe\xff\xff\xff\x03\x00\xfe\xff")



    def test_wav_mono_24_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-24-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        if audio.sample_width == 3:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\xff\xff\x00\x01\x00\x00\xff\xff\x00\x00\x00\x00\x01\x00\x00\xfe\xff\x00\x01\x00\x00\xfe\xff\x00\x04\x00\x00\xfb")
        else:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xfe\xff\x00\x00\x01\x00")

    def test_wav_mono_32_bit(self):
        r = sr.Recognizer()
        audio_file_path = path.join(path.dirname(path.realpath(__file__)), "audio-mono-32-bit-44100Hz.wav")
        with sr.AudioFile(audio_file_path) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 4)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x01\x00\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\xfe\xff\x00\x00\x01\x00")

    def test_wav_stereo_8_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-8-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 1)
        assert_similar(audio.get_raw_data()[:32], b"\x00\xff\x00\xff\x00\x00\xff\x7f\x7f\x00\xff\x00\xff\x00\x00\xff\x00\x7f\x7f\x7f\x00\x00\xff\x00\xff\x00\xff\x00\x7f\x7f\x7f\x7f")

    def test_wav_stereo_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-16-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\x02\x00\xfb\xff\x04\x00\xfe\xff\xfe\xff\x07\x00\xf6\xff\x07\x00\xf9\xff\t\x00\xf5\xff\x0c\x00\xf8\xff\x02\x00\x04\x00\xfa\xff")

    def test_wav_stereo_24_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-24-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        if audio.sample_width == 3:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\xfe\xff\x00\x02\x00\x00\xfe\xff\x00\x00\x00\x00\x02\x00\x00\xfc\xff\x00\x02\x00\x00\xfc\xff\x00\x08\x00\x00\xf6")
        else:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\x00\xfe\xff\x00\x00\x02\x00\x00\x00\xfe\xff\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\xfc\xff\x00\x00\x02\x00")

    def test_wav_stereo_32_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-32-bit-44100Hz.wav")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 4)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\x00\xfe\xff\x00\x00\x02\x00\x00\x00\xfe\xff\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\xfc\xff\x00\x00\x02\x00")

    def test_aiff_mono_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-16-bit-44100Hz.aiff")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\xff\xff\x01\x00\xff\xff\x01\x00\xfe\xff\x02\x00\xfd\xff\x04\x00\xfc\xff\x03\x00\x00\x00\xfe\xff\x03\x00\xfd\xff")

    def test_aiff_stereo_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-16-bit-44100Hz.aiff")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\xfe\xff\x02\x00\xfe\xff\xff\xff\x04\x00\xfa\xff\x04\x00\xfa\xff\t\x00\xf6\xff\n\x00\xfa\xff\xff\xff\x08\x00\xf5\xff")

    def test_flac_mono_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-16-bit-44100Hz.flac")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\x01\x00\xfe\xff\x02\x00\xfc\xff\x06\x00\xf9\xff\x06\x00\xfe\xff\xfe\xff\x05\x00\xfa\xff")

    def test_flac_mono_24_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-mono-24-bit-44100Hz.flac")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        if audio.sample_width == 3:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\xff\xfe\xff\x02\x01\x00\xfd\xfe\xff\x04\x00\x00\xfc\x00\x00\x04\xfe\xff\xfb\x00\x00\x05\xfe\xff\xfc\x03\x00\x04\xfb")
        else:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\xff\xfe\xff\x00\x02\x01\x00\x00\xfd\xfe\xff\x00\x04\x00\x00\x00\xfc\x00\x00\x00\x04\xfe\xff\x00\xfb\x00\x00")

    def test_flac_stereo_16_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-16-bit-44100Hz.flac")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(audio.sample_width, 2)
        assert_similar(audio.get_raw_data()[:32], b"\xff\xff\xff\xff\x02\x00\xfe\xff\x00\x00\x01\x00\xfd\xff\x01\x00\xff\xff\x04\x00\xfa\xff\x05\x00\xff\xff\xfd\xff\x08\x00\xf6\xff")

    def test_flac_stereo_24_bit(self):
        r = sr.Recognizer()
        with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)), "audio-stereo-24-bit-44100Hz.flac")) as source: audio = r.record(source)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(audio.sample_rate, 44100)
        if audio.sample_width == 3:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\xfe\xff\x00\x02\x00\x00\xfe\xff\x00\x00\x00\xff\x01\x00\x02\xfc\xff\xfe\x01\x00\x02\xfc\xff\xfe\x07\x00\x01\xf6")
        else:
            assert_similar(audio.get_raw_data()[:32], b"\x00\x00\x00\x00\x00\x00\xfe\xff\x00\x00\x02\x00\x00\x00\xfe\xff\x00\x00\x00\x00\x00\xff\x01\x00\x00\x02\xfc\xff\x00\xfe\x01\x00")


    # This test tests the invalid file format
    # It should raise OSError
    def test_invalid_audio_file_format(self):
        # Test with an unsupported file format
        r = sr.Recognizer()
        with self.assertRaises(OSError):
            with sr.AudioFile(path.join(path.dirname(path.realpath(__file__)),
                                        "invalid_file.format")) as source:
                audio = r.record(source)

    # This test tests the audio data conversion form 8 to 16 bit
    # input: 8 bit audio data
    #output: 16 bit audio data
    def test_get_raw_data_with_width_conversion_8_to_16_bit(self):
        r = sr.Recognizer()
        audio_file_path = path.join(path.dirname(path.realpath(__file__)), "audio-mono-8-bit-44100Hz.wav")
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
            # Convert 8-bit to 16-bit width
            converted_raw_data = audio.get_raw_data(convert_width=2)
            self.assertIsInstance(converted_raw_data, bytes)
            self.assertEqual(len(converted_raw_data),
                             len(audio.get_raw_data()) * 2)  # 8-bit to 16-bit doubles the data length




    def test_get_raw_data_with_rate_conversion(self):
        r = sr.Recognizer()

        audio_file_path = path.join(path.dirname(path.realpath(__file__)), "audio-mono-16-bit-44100Hz.wav")
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source)
        # Convert to a different sample rate
        converted_raw_data = audio.get_raw_data(convert_rate=22050)
        self.assertIsInstance(converted_raw_data, bytes)
        self.assertNotEqual(len(converted_raw_data),
                            len(audio.get_raw_data()))  # Should differ due to rate conversion

    #Test
    def test_get_raw_data_resample_to_different_rate(self):
        sample_rate = 16000
        target_rate = 8000
        time = 1000/ sample_rate # x01 x02 one sample
        # Create an `AudioData` instance with a sample rate of 16000 Hz
        audio_data = sr.AudioData(b'\x01\x02' * 1000, sample_rate=sample_rate, sample_width=2)

        #Expected length
        expected_length = int (time * target_rate * audio_data.sample_width) # 8000 Hz

        # Convert the audio to a different sample rate 8000 Hz
        result = audio_data.get_raw_data(convert_rate=8000)

        # test result has 8000 Hz
        assert_equal(len(result), expected_length)

        # Verify that the output is in byte format
        self.assertIsInstance(result, bytes)

        # Verify that the output is not empty
        self.assertGreater(len(result), 0)


    # Validate the behavior of the get_raw_data method in audio.py when attempting
    # to convert audio data to 24-bit format on a system that does not natively
    # support 24-bit audio.
    def test_get_raw_data_convert_to_24_bit_no_native_support(self):

        # Create an AudioData object with
        # Raw byte data (b'\x01\x02' * 1000), simulating audio data with repeated byte pairs
        # A sample rate of 16000 Hz
        # A sample width of 2 bytes (16-bit audio)
        audio_data = sr.AudioData(b'\x01\x02' * 1000, sample_rate=16000, sample_width=2)

        # Mocking to simulate lack of 24-bit native support
        # audioop.bias is patched (temporarily replaced) to raise an audioop.error exception
        # This simulates an environment where audioop lacks native support for 24-bit audio
        # Inside the with block, get_raw_data is called with convert_width=3,
        # requesting a 24-bit conversion
        # In the get_raw_data method:
        # Since audioop.bias raises an error, the method is forced to use a fallback method,
        # manually discarding extra bytes to create a 24-bit equivalent.
        # This fallback occurs in cases where direct 24-bit conversion is unsupported.
        with mock.patch('audioop.bias', side_effect=audioop.error):
            result = audio_data.get_raw_data(convert_width=3)

        # These assertions verify:
        # result is of type bytes, indicating that audio data conversion has
        # produced raw audio data in byte format
        # len(result) > 0 ensures that the converted data is non-empty, confirming
        # that the fallback conversion path successfully produced 24-bit data.
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

        # Test audio conversion from 16 to 24 bit
        # Test with existing audio file
        def test_get_raw_data_with_width_conversion_16_to_24_bit(self):
            # recognizer instance
            r = sr.Recognizer()
            audio_file_path = path.join(path.dirname(path.realpath(__file__)), "audio-mono-16-bit-44100Hz.wav")
            with sr.AudioFile(audio_file_path) as source:
                audio = r.record(source)
                # Convert 16-bit to 24-bit width
                converted_raw_data = audio.get_raw_data(convert_width=3)
                self.assertIsInstance(converted_raw_data, bytes)
                self.assertEqual(len(converted_raw_data), len(audio.get_raw_data()) * 3)

    # Following test will test with sample data
    #  Verify the behavior of the get_raw_data() method when converting audio data
    #  to 24-bit format on a system that does support 24-bit audio natively
    def test_get_raw_data_convert_to_24_bit_native_support(self):
        # Set up an AudioData instance
        # Raw byte data (b'\x01\x02' * 1000), simulating audio input with repeated pairs of bytes
        # A sample rate of 16000 Hz
        # A sample width of 2 bytes (indicating 16-bit audio)
        audio_data = sr.AudioData(b'\x01\x02' * 1000, sample_rate=16000, sample_width=2)
        # Convert the audio data to 24-bit format
        # get_raw_data() is called with convert_width=3, which requests a conversion to a 24-bit format
        # In the get_raw_data method:
        # Since audioop natively supports 24-bit audio, audioop.lin2lin will
        # handle the conversion directly from 16-bit to 24-bit
        # The conversion is applied to audio_data.frame_data
        result = audio_data.get_raw_data(convert_width=3)
        # Validate the result
        # Check that result is an instance of bytes, ensuring that the method has
        # produced audio data in byte format after conversion
        # len(result) > 0 confirms that the converted audio data is non-empty
        # validating that the direct conversion to 24-bit format succeeded
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    # Following, tests are not critical but our focus is to compare the result
    # by covering all the (statements, paths, and branches) in the get_raw_data() function
    def test_get_raw_data_convert_to_8_bit_unsigned(self):
        audio_data = sr.AudioData(b'\x01\x02' * 1000, sample_rate=16000, sample_width=2)
        result = audio_data.get_raw_data(convert_width=1)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(0 <= byte <= 255 for byte in result))

    def test_get_raw_data_convert_to_32_bit(self):
        audio_data = sr.AudioData(b'\x01\x02' * 1000, sample_rate=16000, sample_width=2)
        result = audio_data.get_raw_data(convert_width=4)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
