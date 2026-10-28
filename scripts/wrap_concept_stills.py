"""Create unanimated, alpha-preserving still-source movies for Resolve's insert API.
No cutting, transitions, movement, mixing or final delivery is performed here.
"""
from pathlib import Path
import json,sys,subprocess,argparse,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import imageio_ffmpeg
parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
STAGE=Path(args.stage);STAGE.mkdir(parents=True,exist_ok=True)
plan=json.loads((ROOT/'media/concept-film-edit-plan.json').read_text(encoding='utf-8'))
assets={}
for spec in plan['video']:
    name=spec['file']
    if not name.endswith('.mov'):continue
    assets[name]=max(assets.get(name,0),spec['end_s']-spec['start_s']+1)
    if name.startswith('caption-a'):
        index=ord(name[len('caption-a')])-ord('a');source=ROOT/f'media/concept_film_sources/caption-{index:02d}.png'
    elif name=='noto-road.mov':source=ROOT/'media/concept_film_sources/mlit-85-Image199.jpg'
    else:source=ROOT/'media/concept_film_sources'/name.replace('-art.mov','.png')
    shutil.copyfile(source,STAGE/(Path(name).stem+'.png' if source.suffix=='.png' else Path(name).stem+'.jpg'))
for spec in plan['video']:
    if spec['file'].endswith('.mp4'):shutil.copyfile(ROOT/'media/concept_film_sources'/spec['file'],STAGE/spec['file'])
shutil.copyfile(ROOT/'media/purumogu-concept-narration-20261028.mp3',STAGE/'narration.mp3')
shutil.copyfile(ROOT/'media/purumogu-signals-in-the-void.mp3',STAGE/'music.mp3')
def wrap(entry):
    name,duration=entry;dst=STAGE/name;src=dst.with_suffix('.jpg' if name=='noto-road.mov' else '.png')
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-loop','1','-i',str(src),'-t',str(duration),'-r','30','-c:v','qtrle','-pix_fmt','argb','-an',str(dst)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(name+': '+result.stderr)
    print('STATIC_SOURCE',dst.name,flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(wrap,assets.items()))
