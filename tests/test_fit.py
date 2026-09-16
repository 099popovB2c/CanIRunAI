import unittest,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]));import canirunai
class T(unittest.TestCase):
 def test_good(self):
  hw={'ram_gb':16,'gpu':{'vram_gb':8},'disk_free_gb':100};m={'ram_gb':10,'vram_gb':6,'disk_gb':5};self.assertEqual(canirunai.rate(hw,m)[0],'GOOD')
if __name__=='__main__':unittest.main()
