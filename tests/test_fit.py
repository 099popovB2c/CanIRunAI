import unittest,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]));import canirunai
class T(unittest.TestCase):
 def test_estimate_quant(self): self.assertGreater(canirunai.estimate_gb(8,'Q8_0'),canirunai.estimate_gb(8,'Q4_K_M'))
 def test_good(self):
  hw={'ram_gb':16,'vram_gb':8,'disk_free_gb':100};self.assertEqual(canirunai.fit(hw,8,'Q4_K_M')[0],'GOOD')
 def test_override_like_no_gpu(self):
  hw={'ram_gb':16,'vram_gb':0,'disk_free_gb':100};self.assertEqual(canirunai.fit(hw,8,'Q4_K_M')[0],'SLOW')
if __name__=='__main__':unittest.main()
