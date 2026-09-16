import unittest,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]));import canirunai
class T(unittest.TestCase):
 def test_estimate_quant(self): self.assertGreater(canirunai.estimate_gb(8,'Q8_0'),canirunai.estimate_gb(8,'Q4_K_M'))
 def test_good(self):
  hw={'ram_gb':32,'ram_available_gb':32,'vram_gb':12,'disk_free_gb':100};m={'params_b':8,'layers':32,'hidden_size':4096,'attention_heads':32,'kv_heads':8,'context':32768}
  self.assertIn(canirunai.fit_model(hw,m,'Q4_K_M',4096,'fp16')[0],{'GOOD','TIGHT'})
 def test_context_cost(self):
  m={'params_b':8,'layers':32,'hidden_size':4096,'attention_heads':32,'kv_heads':8}
  self.assertGreater(canirunai.kv_cache_gb(m,32768,'fp16'),canirunai.kv_cache_gb(m,4096,'fp16'))
 def test_kv_quant(self):
  m={'params_b':8,'layers':32,'hidden_size':4096,'attention_heads':32,'kv_heads':8}
  self.assertGreater(canirunai.kv_cache_gb(m,8192,'fp16'),canirunai.kv_cache_gb(m,8192,'q4'))
 def test_slow(self):
  hw={'ram_gb':32,'ram_available_gb':28,'vram_gb':0,'disk_free_gb':100};m={'params_b':8}
  self.assertEqual(canirunai.fit_model(hw,m,'Q4_K_M',4096,'fp16')[0],'SLOW')
 def test_speed_none(self): self.assertIsNone(canirunai.speed_estimate({}, {'params_b':70}, 'NO'))
if __name__=='__main__':unittest.main()
