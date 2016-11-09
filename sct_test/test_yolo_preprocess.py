
from ..YOLO.yolo_preprocess import YoloPreprocess

import unittest



class TestPipeLine(unittest.TestCase):
    filepath = '../data_test/vtic_example.txt'
    maplist = ['Rhand', 'ScrewDriver']
    yolo_processor = YoloPreprocess(filepath, maplist=maplist)
    

    def test_df(self):

        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()