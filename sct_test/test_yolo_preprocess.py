import unittest

# Make the YOLO be importable
import sys
sys.path.append('..')
from YOLO.yolo_preprocess import YoloPreprocess


class TestPipeLine(unittest.TestCase):

    file_path = '../data_test/vatic_example.txt'
    maplist = ['Rhand', 'ScrewDriver']
    yoloProcessor = YoloPreprocess(file_path, maplist=maplist)

    def test_df(self):

        self.assertEqual(self.yoloProcessor.get_annotation(1),
            [[0, 359, 204, 69, 83], [1, 257, 100, 64, 94]])

    def test_from_folder(self):
        A = self.yoloProcessor.genYOLO_foler('../data/vatic_id2')
        B = self.yoloProcessor.genYOLO_vid('../data/vatic_id2/output.avi')
        a, b = A.next()
        c, d = B.next()
        self.assertEqual(type(a), type(c)) 

if __name__ == '__main__':
    unittest.main()
