"""First-request sanity checks must accept correct colors in either language."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
from startup_first_requests import output_ok


class FirstRequestTests(unittest.TestCase):
    def test_correct_korean_and_english_color_descriptions(self):
        for label in ('image', 'video'):
            for text in ('처음 빨간색이었다가 파란색으로 바뀝니다.',
                         'The video begins with a solid red screen and then transitions to a solid blue screen.'):
                with self.subTest(label=label, text=text):
                    self.assertTrue(output_ok(label, text, .5, ''))

    def test_wrong_colors_or_missing_output_are_rejected(self):
        for text in ('', 'green then yellow', 'red then green', '파란 화면', 'red and blue\ufffd'):
            self.assertFalse(output_ok('video', text, .5, ''))
        self.assertFalse(output_ok('video', 'red then blue', None, ''))
        self.assertFalse(output_ok('video', 'red then blue', .5, 'unexpected reasoning'))

    def test_text_numbers_are_still_required(self):
        self.assertTrue(output_ok('text', '1, 2, 3, 4, 5', .5, ''))
        self.assertFalse(output_ok('text', '1, 2, 3, 4', .5, ''))


if __name__ == '__main__':
    unittest.main()
