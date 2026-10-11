import os, tempfile, unittest
from pathlib import Path
os.environ['MORPH_API_TOKEN']='test-only-token';os.environ['MORPH_JOBS']=tempfile.mkdtemp();os.environ['MORPH_READY_ENGINES']=''
from fastapi.testclient import TestClient
from server import app
from engines import command
client=TestClient(app);headers={'Authorization':'Bearer test-only-token'}
class ServiceTests(unittest.TestCase):
 def test_auth_and_readiness(self):
  self.assertEqual(client.get('/engines').status_code,401)
  result=client.get('/engines',headers=headers);self.assertEqual(result.status_code,200)
  self.assertEqual(set(result.json()),{'deca','instantid','liveportrait','liveportrait-animal','facefusion','unum','text-image'})
  self.assertFalse(any(x['ready'] for x in result.json().values()))
 def test_uninstalled_engine_cannot_accept_photo(self):
  response=client.post('/jobs',headers=headers,json={'engine':'instantid','photo':'not-an-image','params':{'prompt':'a cartoon'}})
  self.assertEqual(response.status_code,409)
 def test_unknown_engine_and_unlisted_output(self):
  self.assertEqual(client.post('/jobs',headers=headers,json={'engine':'shell'}).status_code,400)
  self.assertEqual(client.get('/jobs/not-a-job/files/etc/passwd',headers=headers).status_code,404)
 def test_invalid_source_identity_is_rejected_before_queuing(self):
  from unittest.mock import patch
  with patch.dict(os.environ,{'SDXL_MODEL':tempfile.mkdtemp(),'MORPH_READY_ENGINES':'text-image'}):
   response=client.post('/jobs',headers=headers,json={'engine':'text-image','params':{'prompt':'a generated form','identity_id':'wrong-source'}})
   self.assertEqual(response.status_code,400)
   self.assertIn('identity reference',response.json()['detail'])
 def test_prompt_is_one_argument_not_shell_code(self):
  root=Path(tempfile.mkdtemp());os.environ['INSTANTID_ROOT']=str(root);job=root/'job';job.mkdir()
  prompt='a purple unicorn; $(touch /tmp/should-not-exist)'
  args,cwd=command('instantid',job,{'prompt':prompt})
  self.assertEqual(args[args.index('--prompt')+1],prompt);self.assertEqual(cwd,root)
if __name__=='__main__':unittest.main()
