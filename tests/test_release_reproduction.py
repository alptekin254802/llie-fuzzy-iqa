"""Check format fallback rejection and invalid LMM logits without model inference."""
import unittest
from unittest.mock import patch
import cv2
from realtime import camera
from analysis.lmm_extra.scoring import quality

class Camera:
    def __init__(self,fallback=False):self.values={};self.requests=[];self.fallback=fallback;self.released=False
    def isOpened(self):return True
    def getBackendName(self):return 'DSHOW'
    def set(self,key,value):self.requests.append(key);self.values[key]=value;return True
    def get(self,key):
        if key==cv2.CAP_PROP_AUTO_EXPOSURE:return -1
        if key==cv2.CAP_PROP_AUTOFOCUS:return 2
        if key==cv2.CAP_PROP_FOURCC and self.fallback:return cv2.VideoWriter_fourcc(*'YUY2')
        return self.values[key]
    def release(self):self.released=True

class ReleaseReproductionTests(unittest.TestCase):
    def test_format_requested_after_dimensions_and_fps(self):
        cap=Camera()
        with patch.object(camera.cv2,'VideoCapture',return_value=cap),patch.object(camera,'read'):
            self.assertIs(camera.open_camera(),cap)
        self.assertEqual(cap.requests[:4],[cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FOURCC])
        self.assertEqual(cap.get(cv2.CAP_PROP_EXPOSURE),-6)
    def test_format_fallback_aborts_and_releases_camera(self):
        cap=Camera(fallback=True)
        with patch.object(camera.cv2,'VideoCapture',return_value=cap),patch.object(camera,'read'):
            with self.assertRaisesRegex(RuntimeError,'format readback'):camera.open_camera()
        self.assertTrue(cap.released)
    def test_quality_rejects_nonfinite_and_wrong_token_count(self):
        for logits in [[1,2,3,4],[1,2,float('nan'),4,5],[1,2,3,4,float('inf')]]:
            with self.assertRaises(FloatingPointError):quality(logits)
        self.assertAlmostEqual(quality([0,0,0,0,0]),.5)
