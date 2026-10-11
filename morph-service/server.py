"""Optional model host. Local photo-only operations do not call this service."""
import base64, io, json, os, uuid, threading, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from engines import ROOTS, readiness, execute

app=FastAPI();app.add_middleware(CORSMiddleware,allow_origins=os.environ.get('MORPH_ORIGINS','http://localhost:8765').split(','),allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
DATA=Path(os.environ.get('MORPH_JOBS','./jobs')).resolve();DATA.mkdir(parents=True,exist_ok=True)
queue=ThreadPoolExecutor(max_workers=1);jobs={};lock=threading.Lock()
TOKEN=os.environ.get('MORPH_API_TOKEN')
if not TOKEN:raise RuntimeError('Set MORPH_API_TOKEN before starting the model service.')

def auth(request:Request):
 import secrets
 if not secrets.compare_digest(request.headers.get('authorization',''),f'Bearer {TOKEN}'):raise HTTPException(401,'Model service authentication required')

def asset(value,path,image=True):
 if not isinstance(value,str) or len(value)>28_000_000:raise ValueError('Upload too large or missing')
 raw=base64.b64decode(value.split(',',1)[-1],validate=True)
 if len(raw)>20_000_000:raise ValueError('Upload exceeds 20 MB')
 if image:
  with Image.open(io.BytesIO(raw)) as photo:
   if photo.width*photo.height>20_000_000:raise ValueError('Image exceeds 20 megapixels')
   photo.convert('RGB').save(path)
 else:path.write_bytes(raw)

def run(id,engine,path,params):
 jobs[id]['status']='running'
 try:
  files=execute(engine,path,params);jobs[id].update(status='done',files=[str(p.relative_to(path)) for p in files])
 except Exception as error:jobs[id].update(status='failed',error=str(error),files=['worker.log'] if (path/'worker.log').exists() else [])
 (path/'receipt.json').write_text(json.dumps(jobs[id]))

@app.get('/engines',dependencies=[Depends(auth)])
def engines():return {engine:readiness(engine) for engine in [*ROOTS,'text-image']}

@app.post('/jobs',dependencies=[Depends(auth)])
async def submit(request:Request):
 # Bound transport memory before decoding JSON/base64.
 body=bytearray()
 async for chunk in request.stream():
  body.extend(chunk)
  if len(body)>55_000_000:raise HTTPException(413,'Request too large')
 try:
  data=json.loads(body);engine=data['engine'];params=data.get('params',{})
  if engine not in [*ROOTS,'text-image']:raise ValueError('Unknown engine')
  if not readiness(engine)['ready']:raise HTTPException(409,readiness(engine)['reason'])
  if not isinstance(params,dict):raise ValueError('Invalid parameters')
  if engine in ['instantid','text-image'] and (not isinstance(params.get('prompt'),str) or not 1<=len(params['prompt'])<=4000):raise ValueError('Describe the requested picture')
  seed=params.get('seed',0)
  if not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('Invalid seed')
  with lock:
   if sum(j['status'] in ['queued','running'] for j in jobs.values())>=4:raise HTTPException(429,'Model queue full; retry after a job finishes')
   id=uuid.uuid4().hex;path=DATA/id;path.mkdir()
   if engine!='text-image':asset(data.get('photo'),path/'photo.png')
   if engine.startswith('liveportrait'):asset(data.get('driving'),path/'driving.mp4',False)
   if engine=='facefusion':asset(data.get('target'),path/'target.png')
   jobs[id]={'id':id,'engine':engine,'status':'queued','files':[],'params':params,'identity_sha256':hashlib.sha256((path/'photo.png').read_bytes()).hexdigest() if (path/'photo.png').exists() else None}
   queue.submit(run,id,engine,path,params)
  return {'id':id}
 except HTTPException:raise
 except (ValueError,KeyError,TypeError,OSError) as error:raise HTTPException(400,str(error))

@app.get('/jobs/{id}',dependencies=[Depends(auth)])
def status(id:str):
 if id not in jobs:raise HTTPException(404,'Job not found in this service session')
 return jobs[id]

@app.get('/jobs/{id}/files/{name:path}',dependencies=[Depends(auth)])
def output(id:str,name:str):
 if id not in jobs or name not in jobs[id]['files']:raise HTTPException(404,'Output not found')
 path=(DATA/id/name).resolve()
 if not path.is_relative_to(DATA/id):raise HTTPException(400,'Invalid output path')
 return FileResponse(path,filename=path.name)
