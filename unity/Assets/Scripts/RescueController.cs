using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed class RescueController : MonoBehaviour
    {
        RescueSimulation simulation;
        RescueMapView map;
        readonly RescueEvaluation evaluation=new RescueEvaluation();
        UnityRustBridge bridge;
        Camera physicalCamera,mapCamera,robotCamera;
        Font font;
        int sceneIndex=0,seed=20261027,port=47120,selectedNode,maxSteps=9000;
        long sequence,expected=-1;
        string run,status="READY",failure="";
        bool running,paused,waiting,scenePicker,cutaway=true,showTruth,closeup;
        float accumulator,lastSent,lastReply,latency,entropy=1,speed=1;
        JObject lastMap;
        readonly List<string> audit=new List<string>();
        bool record;string frameFolder;int frame,maxFrames=1400;float captureAt;
        readonly List<string> frameTimes=new List<string>{"frame_index,wall_time_s,sim_step"};
        static readonly Color Background=new Color(.035f,.055f,.07f),Panel=new Color(.065f,.09f,.108f),Ink=new Color(.96f,.98f,.98f),Muted=new Color(.76f,.85f,.87f),Mint=new Color(.4f,1,.84f),Amber=new Color(1,.76f,.44f);
        static readonly Rect WorldArea=new Rect(24,166,770,406),MapArea=new Rect(810,166,766,406);

        void Start()
        {
            Application.runInBackground=true;Application.targetFrameRate=60;font=Font.CreateDynamicFontFromOSFont(new[]{"Yu Gothic","Segoe UI"},18);
            var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++){if(args[i]=="--closeup")closeup=args[i+1]=="true";if(args[i]=="--port")int.TryParse(args[i+1],out port);if(args[i]=="--steps")int.TryParse(args[i+1],out maxSteps);if(args[i]=="--seed")int.TryParse(args[i+1],out seed);if(args[i]=="--scene")int.TryParse(args[i+1],out sceneIndex);if(args[i]=="--frames"){record=true;frameFolder=args[i+1];Directory.CreateDirectory(frameFolder);}if(args[i]=="--capture-count")int.TryParse(args[i+1],out maxFrames);if(args[i]=="--speed")float.TryParse(args[i+1],NumberStyles.Float,CultureInfo.InvariantCulture,out speed);}
            maxSteps=Mathf.Clamp(maxSteps,500,30000);sceneIndex=Mathf.Clamp(sceneIndex,0,14);Build();if(record)Begin();
        }
        void Build()
        {
            bridge?.Dispose();bridge=null;simulation?.Dispose();map?.Dispose();
            simulation=new RescueSimulation(sceneIndex,seed);map=new RescueMapView();lastMap=null;expected=-1;waiting=false;running=false;failure="";status="READY";audit.Clear();
            if(!physicalCamera)
            {
                physicalCamera=new GameObject("PhysicalSimulationCamera").AddComponent<Camera>();physicalCamera.clearFlags=CameraClearFlags.SolidColor;physicalCamera.backgroundColor=new Color(.17f,.21f,.22f);physicalCamera.cullingMask=~(1<<8);physicalCamera.orthographic=true;physicalCamera.nearClipPlane=.005f;physicalCamera.farClipPlane=30;
                mapCamera=new GameObject("InferenceOnlyCamera").AddComponent<Camera>();mapCamera.clearFlags=CameraClearFlags.SolidColor;mapCamera.backgroundColor=new Color(.025f,.055f,.075f);mapCamera.cullingMask=1<<8;mapCamera.orthographic=true;mapCamera.nearClipPlane=.005f;mapCamera.farClipPlane=30;mapCamera.depth=1;
                robotCamera=new GameObject("ArticulatedMechanismDetail").AddComponent<Camera>();robotCamera.clearFlags=CameraClearFlags.SolidColor;robotCamera.backgroundColor=new Color(.12f,.17f,.19f);robotCamera.cullingMask=1<<9;robotCamera.orthographic=true;robotCamera.orthographicSize=.055f;robotCamera.nearClipPlane=.002f;robotCamera.farClipPlane=2;robotCamera.depth=2;robotCamera.rect=Viewport(new Rect(38,416,240,146));
                var light=new GameObject("RescueDaylight").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.7f;light.shadows=LightShadows.Soft;light.transform.rotation=Quaternion.Euler(48,-28,0);RenderSettings.ambientMode=UnityEngine.Rendering.AmbientMode.Flat;RenderSettings.ambientLight=new Color(.52f,.60f,.64f);
            }
            physicalCamera.rect=Viewport(WorldArea);mapCamera.rect=Viewport(MapArea);UpdateCameras();simulation.Scene.Cutaway(cutaway);simulation.Scene.ShowTruth(showTruth);
        }
        static Rect Viewport(Rect r)=>new Rect(r.x/1600,(900-r.y-r.height)/900,r.width/1600,r.height/900);
        void Begin()
        {
            if(simulation.Step>0)Build();run=Guid.NewGuid().ToString("N");sequence=0;running=true;paused=false;status="CONNECTING";
            bridge=new UnityRustBridge{MaxFrameBytes=8388608};bridge.Start(port);lastReply=Time.realtimeSinceStartup;
        }
        void Update()
        {
            if(running)
            {
                try
                {
                    if(bridge.Failure!=null)throw new IOException(bridge.Failure);
                    while(bridge.Incoming.TryDequeue(out JObject reply))
                    {
                        if((string)reply["type"]=="error")throw new IOException((string)reply["reason"]);
                        if((string)reply["protocol_version"]!="2.0"||(string)reply["run_id"]!=run||(long)reply["seq_no"]!=expected)throw new IOException("map response contract mismatch");
                        waiting=false;lastReply=Time.realtimeSinceStartup;latency=(lastReply-lastSent)*1000;lastMap=reply;entropy=(float)reply["mean_entropy_bits"];map.Update(reply);simulation.ApplyGuidance(reply);evaluation.Evaluate(simulation,reply);status=paused?"PAUSED":"EXPLORING";
                    }
                    if(Time.realtimeSinceStartup-lastReply>15)throw new IOException("Rust mapping timed out");
                    if(bridge.Connected)
                    {
                        // One batch in flight; physics keeps running with the last safe target.
                        if(!paused&&simulation.Step<maxSteps)
                        {
                            accumulator+=Time.deltaTime*speed;int count=0;
                            while(accumulator>=.02f&&count++<12&&simulation.Step<maxSteps)
                            {
                                accumulator-=.02f;simulation.Tick();
                                if(simulation.Step%25==0&&!waiting)
                                {
                                    var packet=simulation.Observe(run,sequence++);string json=JsonConvert.SerializeObject(packet);bridge.SendJson(json);audit.Add(json);expected=packet.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;
                                }
                            }
                        }
                        if(paused&&!waiting&&Time.realtimeSinceStartup-lastSent>5)
                        {
                            var heartbeat=new RescuePacket{run_id=run,seq_no=sequence++,sim_step=simulation.Step};string json=JsonConvert.SerializeObject(heartbeat);bridge.SendJson(json);audit.Add(json);expected=heartbeat.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;
                        }
                    }
                    if(simulation.Step>=maxSteps)
                    {
                        simulation.Power(false);
                        if(!waiting&&(lastMap==null||(long)lastMap["sim_step"]<maxSteps)){var packet=simulation.Observe(run,sequence++);string json=JsonConvert.SerializeObject(packet);bridge.SendJson(json);audit.Add(json);expected=packet.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;}
                        if(!waiting&&lastMap!=null&&(long)lastMap["sim_step"]>=maxSteps){running=false;status="TIME BUDGET COMPLETE";Export();}
                    }
                }
                catch(Exception e){failure=e.Message;status="SAFE STOP";running=false;simulation.Power(false);simulation.ReleaseAll();bridge?.Dispose();bridge=null;Export();}
            }
            UpdateCameras();
            if(record&&Time.realtimeSinceStartup>=captureAt)
            {
                captureAt=Time.realtimeSinceStartup+.05f;frameTimes.Add(frame+","+Time.realtimeSinceStartup.ToString("F5",CultureInfo.InvariantCulture)+","+simulation.Step);ScreenCapture.CaptureScreenshot(Path.Combine(frameFolder,$"frame-{frame++:D5}.png"));if(frame>=maxFrames){Export();Application.Quit();}
            }
        }
        void UpdateCameras()
        {
            if(!physicalCamera||simulation==null)return;
            if(closeup){var target=simulation.Robots[Mathf.Clamp(selectedNode,0,simulation.Robots.Count-1)].Core.position;physicalCamera.transform.position=target+new Vector3(.085f,.065f,-.065f);physicalCamera.transform.LookAt(target);physicalCamera.orthographicSize=.072f;}
            else{physicalCamera.transform.position=new Vector3(2.05f,2.0f,-.6f);physicalCamera.transform.LookAt(new Vector3(0,.06f,1.30f));physicalCamera.orthographicSize=1.12f;}
            float orbit=Time.realtimeSinceStartup*.025f;var focus=new Vector3(0,.18f,1.25f);mapCamera.transform.position=focus+new Vector3(Mathf.Cos(orbit)*2.9f,2.1f,-Mathf.Sin(orbit)*.7f-2.4f);mapCamera.transform.LookAt(focus);mapCamera.orthographicSize=1.20f;
            var robotTarget=simulation.Robots[Mathf.Clamp(selectedNode,0,simulation.Robots.Count-1)].Core.position;robotCamera.enabled=!closeup;robotCamera.transform.position=robotTarget+new Vector3(.060f,.055f,-.065f);robotCamera.transform.LookAt(robotTarget);
            for(int i=0;i<simulation.Robots.Count;i++)foreach(var renderer in simulation.Robots[i].Root.GetComponentsInChildren<Renderer>())renderer.gameObject.layer=i==selectedNode?9:0;
        }
        void Export()
        {
            if(audit.Count==0)return;string path=Path.Combine(Application.persistentDataPath,"rescue-runs");Directory.CreateDirectory(path);var output=Path.Combine(path,run+".jsonl");File.WriteAllLines(output,audit);if(lastMap!=null)File.AppendAllText(output,lastMap.ToString(Formatting.None)+Environment.NewLine);File.WriteAllText(Path.Combine(path,run+"-evaluation.json"),JsonConvert.SerializeObject(evaluation.Snapshot(simulation),Formatting.Indented));
        }
        void OnDestroy(){if(record&&frameFolder!=null)File.WriteAllLines(Path.Combine(frameFolder,"timestamps.csv"),frameTimes);bridge?.Dispose();simulation?.Dispose();map?.Dispose();}
        void Fill(Rect r,Color c){GUI.color=c;GUI.DrawTexture(r,Texture2D.whiteTexture);GUI.color=Color.white;}
        void Text(float x,float y,float w,float h,string value,int size,Color color,bool bold=false)
        {
            var style=new GUIStyle(GUI.skin.label){font=font,fontSize=size,fontStyle=bold?FontStyle.Bold:FontStyle.Normal,wordWrap=true};style.normal.textColor=color;GUI.Label(new Rect(x,y,w,h),value,style);
        }
        bool Button(float x,float y,float width,string label,bool selected=false,float height=30)
        {
            Fill(new Rect(x,y,width,height),selected?new Color(.12f,.32f,.29f):new Color(.12f,.17f,.20f));var style=new GUIStyle(GUI.skin.button){font=font,fontSize=13};style.normal.background=null;style.hover.background=null;style.active.background=null;style.normal.textColor=selected?Mint:Ink;return GUI.Button(new Rect(x,y,width,height),label,style);
        }
        void Header(float x,float y,float width,string number,string title,string note)
        {
            Fill(new Rect(x,y,width,38),Panel);Text(x+12,y+8,28,28,number,12,Mint,true);Text(x+46,y+6,width-100,28,title,18,Ink,true);Text(x+12,y+452,width-24,30,note,12,Muted);
        }
        void OnGUI()
        {
            if(simulation==null)return;GUI.matrix=Matrix4x4.Scale(new Vector3(Screen.width/1600f,Screen.height/900f,1));
            Fill(new Rect(0,0,1600,122),Background);Fill(new Rect(0,572,1600,328),Background);Fill(new Rect(0,122,24,450),Background);Fill(new Rect(794,122,16,450),Background);Fill(new Rect(1576,122,24,450),Background);
            Text(24,16,550,38,"PURUMOGU / 災害空隙サーチ",27,Ink,true);Text(26,56,750,25,"7 cm articulated swarm · sensor-only 3D reconstruction · simulation",13,Muted);
            var spec=RescueScene.Catalog[sceneIndex];
            if(Button(24,88,430,$"{spec.Id}  {spec.Name}  /  {new[]{"易","中","難"}[spec.Difficulty]}",scenePicker))scenePicker=!scenePicker;
            Text(470,91,630,28,spec.Disaster+" / "+spec.Feature,14,Muted);
            if(Button(1030,18,120,running?(paused?"再開":"一時停止"):"探索開始",running)){if(running)paused=!paused;else Begin();}
            if(Button(1160,18,100,"リセット"))Build();
            if(Button(1270,18,130,"全接続を解除"))simulation.ReleaseAll();
            if(Button(1410,18,166,"自動接続 "+(simulation.Docking?"ON":"OFF"),simulation.Docking)){simulation.Docking=!simulation.Docking;if(!simulation.Docking)simulation.ReleaseAll();}
            if(Button(1030,59,120,"機体拡大",closeup))closeup=!closeup;
            if(Button(1160,59,100,"断面表示",cutaway)){cutaway=!cutaway;simulation.Scene.Cutaway(cutaway);}
            if(Button(1270,59,130,"評価用の真値",showTruth)){showTruth=!showTruth;simulation.Scene.ShowTruth(showTruth);}
            if(Button(1410,59,80,$"{speed:F0}×"))speed=speed<2?4:1;
            if(Button(1500,59,76,"保存"))Export();
            Header(24,126,770,"01","ロボットと災害現場",$"{status}  ·  {simulation.Step*.02f:F1} s  ·  接続 {simulation.Links.Count} / 累計 {simulation.Attached}  ·  解除 {simulation.Detached}  ·  断面は描画のみ");
            Header(810,126,766,"02","観測とともに復元する3Dマップ",$"青緑：観測支持 {map.VisibleObservedCells}  /  紫：周辺補完 {map.VisibleCompletionCells}  /  暗部：未表示・未知  /  橙：音響候補");
            if(lastMap==null)Text(850,310,660,90,"観測待ち\n正解の地形は、この画面へ転送しません。",21,Muted);
            for(int i=0;i<map.CuePositions.Count;i++)
            {
                var p=mapCamera.WorldToViewportPoint(map.CuePositions[i]);float pinX=MapArea.x+p.x*MapArea.width,pinY=MapArea.y+(1-p.y)*MapArea.height;Text(pinX,pinY,32,28,((char)('A'+i)).ToString(),18,Amber,true);
                float x=MapArea.xMax-202,y=180+i*89;Fill(new Rect(x,y,195,80),new Color(.06f,.10f,.12f,.95f));Text(x+8,y+5,182,72,map.CueLabels[i],13,Amber,true);
            }
            if(lastMap!=null&&map.CuePositions.Count==0)Text(832,180,700,30,"要救助者候補：保留 / 独立した観測の一致がまだ不足",14,Amber);
            Fill(new Rect(24,620,770,256),Panel);Fill(new Rect(810,620,766,256),Panel);
            Text(38,630,730,26,"03  各ノードの状態",18,Ink,true);Text(42,666,730,22,"NODE    電池      関節 / 回転     接続    受信peer    RSSI     音響SNR    Δ温度    状態",12,Muted,true);
            for(int i=0;i<simulation.Robots.Count;i++)
            {
                var node=simulation.Robots[i];float y=692+i*14.3f;if(i==selectedNode)Fill(new Rect(36,y,744,15),new Color(.10f,.20f,.22f));
                string row=$"{node.Id}     {node.Energy*100,3:F0}%     {node.BendDegrees,3:F0}°/{node.SpinDegrees,3:F0}°     {(node.PortBusy[0]?1:0)+(node.PortBusy[1]?1:0)}        {node.Peers,2}          {node.Signal,4:F0}       {node.LastSnr,4:F1}       {node.Temperature,4:F1}°     {(node.Lost?"喪失":node.Verifying?"確認中":node.Stuck?"停滞":node.Powered?"稼働":"停止")}";
                Text(42,y,730,20,row,11,node.Lost||node.Stuck?Amber:Ink);if(GUI.Button(new Rect(36,y,744,15),GUIContent.none,GUIStyle.none))selectedNode=i;
            }
            Text(826,630,730,26,"04  推論の評価 / 真値はこの区画だけで使用",18,Ink,true);
            float humanPrecision=evaluation.CueCount>0?(float)evaluation.Matched/evaluation.CueCount:0,humanRecall=simulation.Scene.Spec.Victims>0?(float)evaluation.Matched/simulation.Scene.Spec.Victims:0;
            Metric(832,670,"3D占有IoU / 全領域",$"{evaluation.GlobalIoU*100:F1}%",evaluation.GlobalIoU>=.95f);Metric(1070,670,"要救助者候補の適合率",evaluation.CueCount>0?$"{humanPrecision*100:F1}%":"候補なし",humanPrecision>=.95f);Metric(1320,670,"要救助者候補の再現率",simulation.Scene.Spec.Victims>0?$"{humanRecall*100:F1}%":"負例",humanRecall>=.95f);
            Text(834,745,700,23,$"全推定地図を評価（右上は観測の進行表示） / 観測率 {evaluation.Coverage*100:F1}% · 既観測IoU {evaluation.IoU:F2} · 位置RMSE {evaluation.PoseRmse*100:F1} cm",13,Ink);
            Text(834,777,700,24,$"候補 {evaluation.CueCount} / 一致 {evaluation.Matched} / 誤警報 {evaluation.FalseAlarms} · 位置誤差 {(float.IsNaN(evaluation.LocalizationError)?"未評価":(evaluation.LocalizationError*100).ToString("F1")+" cm")} · Rust {latency:F0} ms",13,Muted);
            Text(834,810,700,48,string.IsNullOrEmpty(failure)?"評価対象は実験用の局所空隙です。水・粉塵・音響は簡略モデル。実災害の救助性能を保証する検証ではありません。":failure,12,string.IsNullOrEmpty(failure)?Muted:Amber);
            if(scenePicker)
            {
                Fill(new Rect(24,125,750,460),new Color(.045f,.075f,.09f,.99f));Text(40,138,710,28,"15 SCENES / 易・中・難 各5",18,Mint,true);
                for(int i=0;i<15;i++){var item=RescueScene.Catalog[i];if(Button(40,177+i*25,718,$"{item.Id}  {item.Name}  /  {item.Disaster}  ·  {item.Feature}",i==sceneIndex,24)){sceneIndex=i;scenePicker=false;Build();}}
            }
        }
        void Metric(float x,float y,string label,string value,bool passed){Text(x,y,225,24,label,12,Muted);Text(x,y+24,225,42,value,30,passed?Mint:Amber,true);}
    }
}
