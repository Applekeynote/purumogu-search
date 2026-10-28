"""Read-only QC of the actual Resolve export; never edits or remixes the deliverable."""
from pathlib import Path
import sys,json,hashlib,re,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import imageio_ffmpeg
from PIL import Image,ImageDraw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('video');args=parser.parse_args();video=Path(args.video)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);reader.close()
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
    assert meta['size']==(1920,1080),meta
    assert abs(meta['fps']-30)<.001 and frames==4485 and abs(seconds-149.5)<.1,(frames,seconds)
    qc=subprocess.run([ffmpeg,'-hide_banner','-nostats','-i',str(video),'-vf','blackdetect=d=0.2:pix_th=0.015:pic_th=0.999','-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    black=[dict(start=float(a),end=float(b),duration=float(c)) for a,b,c in re.findall(r'black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)',qc.stderr)]
    loudness=float(re.findall(r'I:\s+(-?[\d.]+) LUFS',qc.stderr)[-1]);peak=float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS',qc.stderr)[-1])
    assert -21<loudness<-14 and peak<0,(loudness,peak)
    interior=[b for b in black if b['start']>1 and b['end']<148.5]
    assert not interior,interior
    times=[6,15,24,34,43,49,52,59,69,75,80,86,95,107,115,127,136,144]
    sheet=Image.new('RGB',(1440,6*294),'#070b10');draw=ImageDraw.Draw(sheet)
    for i,t in enumerate(times):
        data=subprocess.run([ffmpeg,'-v','error','-ss',str(t),'-i',str(video),'-frames:v','1','-vf','scale=480:270','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout
        frame=Image.frombytes('RGB',(480,270),data);x=(i%3)*480;y=(i//3)*294;sheet.paste(frame,(x,y));draw.text((x+12,y+273),f'{t//60:02}:{t%60:02}',fill='white')
        if t==49:
            full=subprocess.run([ffmpeg,'-v','error','-ss',str(t),'-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True).stdout
            (ROOT/'media/purumogu-concept-poster.png').write_bytes(full)
    sheet.save(ROOT/'tmp/concept-film-contact-sheet.png')
    result=dict(file='media/purumogu-concept-film.mp4',editor='DaVinci Resolve Studio 21.1',timeline='Signals_in_the_Void_MASTER',sha256=hashlib.sha256(video.read_bytes()).hexdigest(),bytes=video.stat().st_size,size=meta['size'],fps=meta['fps'],frames=frames,duration_s=seconds,integrated_lufs=loudness,true_peak_dbfs=peak,black_intervals=black,representative_frames=times,checks='All frames decoded; exact planned duration; no interior essentially empty black interval >=0.2s; unclipped audio; rendered visual samples require human/agent review')
    (ROOT/'reports/concept-film-qc.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
