using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueController : MonoBehaviour
    {
        RescueSimulation simulation;
        RescueEchoWireMap map;
        readonly RescueEvaluation evaluation=new RescueEvaluation();
        UnityRustBridge bridge;
        BundledInference bundled;
        bool externalInference;
        int sourceSeed;
        Camera physicalCamera,mapCamera,robotCamera;
        Font font;
        int sceneIndex=15,seed=20261027,port=47120,selectedNode,maxSteps=18000;
        int nodeCount=24,deploymentSeed=907231;
        bool entranceDeployment;
        Vector2 nodeScroll,sceneScroll;
        readonly RescueViewport physicalView=new RescueViewport(),inferenceView=new RescueViewport();
        string utilityMenu="";
        long sequence,expected=-1;
        string run,status="READY",failure="";
        bool running,paused,waiting,scenePicker,cutaway=true,showTruth=true,closeup;
        float accumulator,lastSent,lastReply,latency,entropy=1,speed=1;
        JObject lastMap;
        readonly List<string> audit=new List<string>();
        bool record;string frameFolder;int frame,maxFrames=1400;float captureAt;
        string filmView="";
        readonly List<string> frameTimes=new List<string>{"frame_index,wall_time_s,sim_step"};
        static readonly Color Background=new Color(.035f,.055f,.07f),Panel=new Color(.065f,.09f,.108f),Ink=new Color(.96f,.98f,.98f),Muted=new Color(.76f,.85f,.87f),Mint=new Color(.4f,1,.84f),Amber=new Color(1,.76f,.44f);
        static readonly Rect WorldArea=new Rect(24,166,770,406),MapArea=new Rect(810,166,766,406);

        void Start()
        {
            Application.runInBackground=true;QualitySettings.antiAliasing=4;Application.targetFrameRate=60;font=Font.CreateDynamicFontFromOSFont(new[]{"Yu Gothic","Segoe UI"},18);
            var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++){if(args[i]=="--closeup")closeup=args[i+1]=="true";if(args[i]=="--port")int.TryParse(args[i+1],out port);if(args[i]=="--steps")int.TryParse(args[i+1],out maxSteps);if(args[i]=="--seed")int.TryParse(args[i+1],out seed);if(args[i]=="--scene")int.TryParse(args[i+1],out sceneIndex);if(args[i]=="--frames"){record=true;frameFolder=args[i+1];Directory.CreateDirectory(frameFolder);}if(args[i]=="--capture-count")int.TryParse(args[i+1],out maxFrames);if(args[i]=="--speed")float.TryParse(args[i+1],NumberStyles.Float,CultureInfo.InvariantCulture,out speed);}
            for(int i=0;i<args.Length-1;i++){if(args[i]=="--nodes")int.TryParse(args[i+1],out nodeCount);if(args[i]=="--deployment-seed")int.TryParse(args[i+1],out deploymentSeed);if(args[i]=="--deployment")entranceDeployment=args[i+1]=="entrance";if(args[i]=="--film-view")filmView=args[i+1];}
            nodeCount=nodeCount==36?36:nodeCount==48?48:entranceDeployment&&nodeCount==12?12:24;
            sourceSeed=Guid.NewGuid().GetHashCode();
            for(int i=0;i<args.Length-1;i++){if(args[i]=="--seed")sourceSeed=seed;if(args[i]=="--port")externalInference=true;}
            for(int i=0;i<args.Length-1;i++)if(args[i]=="--source-seed")int.TryParse(args[i+1],out sourceSeed);
            maxSteps=Mathf.Clamp(maxSteps,500,30000);sceneIndex=Mathf.Clamp(sceneIndex,0,RescueScene.Catalog.Length-1);Build();
            if(!externalInference)
            {
                try {bundled=new BundledInference();bundled.Start(Path.Combine(Application.streamingAssetsPath,"inference"));port=bundled.Port;}
                catch(Exception e){failure=e.Message;status="INFERENCE UNAVAILABLE";}
            }
            Debug.Log($"RESCUE_START nodes={nodeCount} cameras={Camera.allCamerasCount} film={filmView??"<null>"} gui={useGUILayout} inference={(externalInference?"external":bundled!=null&&bundled.Alive?"bundled":"unavailable")}");
            if(record&&failure=="")Begin();
        }
        void Build()
        {
            bridge?.Dispose();bridge=null;simulation?.Dispose();map?.Dispose();
            simulation=new RescueSimulation(sceneIndex,seed,nodeCount,entranceDeployment?RescueDeploymentMode.EntranceGrid:RescueDeploymentMode.WholeSceneDrop,deploymentSeed,sourceSeed);map=new RescueEchoWireMap();lastMap=null;expected=-1;waiting=false;running=false;failure="";status="READY";audit.Clear();selectedNode=Mathf.Clamp(selectedNode,0,nodeCount-1);nodeScroll=Vector2.zero;
            if(!physicalCamera)
            {
                physicalCamera=new GameObject("PhysicalSimulationCamera").AddComponent<Camera>();physicalCamera.clearFlags=CameraClearFlags.SolidColor;physicalCamera.backgroundColor=new Color(.17f,.21f,.22f);physicalCamera.cullingMask=~(1<<8);physicalCamera.orthographic=true;physicalCamera.nearClipPlane=.005f;physicalCamera.farClipPlane=30;
                mapCamera=new GameObject("InferenceOnlyCamera").AddComponent<Camera>();mapCamera.clearFlags=CameraClearFlags.SolidColor;mapCamera.backgroundColor=new Color(.004f,.008f,.014f);mapCamera.cullingMask=1<<8;mapCamera.orthographic=true;mapCamera.nearClipPlane=.005f;mapCamera.farClipPlane=30;mapCamera.depth=1;
                robotCamera=new GameObject("ArticulatedMechanismDetail").AddComponent<Camera>();robotCamera.clearFlags=CameraClearFlags.SolidColor;robotCamera.backgroundColor=new Color(.12f,.17f,.19f);robotCamera.cullingMask=1<<9;robotCamera.orthographic=true;robotCamera.orthographicSize=.055f;robotCamera.nearClipPlane=.002f;robotCamera.farClipPlane=2;robotCamera.depth=2;robotCamera.rect=Viewport(new Rect(38,416,240,146));
                var light=new GameObject("RescueDaylight").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.7f;light.shadows=LightShadows.Soft;light.transform.rotation=Quaternion.Euler(48,-28,0);RenderSettings.ambientMode=UnityEngine.Rendering.AmbientMode.Flat;RenderSettings.ambientLight=new Color(.52f,.60f,.64f);
            }
            physicalCamera.rect=Viewport(WorldArea);mapCamera.rect=Viewport(MapRenderArea);UpdateCameras();simulation.Scene.Cutaway(cutaway);simulation.Scene.ShowTruth(showTruth);
        }
        static Rect Viewport(Rect r)=>new Rect(r.x/1600,(900-r.y-r.height)/900,r.width/1600,r.height/900);
        void Begin()
        {
            if(!externalInference&&(bundled==null||!bundled.Alive)){failure="同梱の推論プロセスを起動できません。アプリを再起動し、ZIPの展開先を確認してください。";status="INFERENCE UNAVAILABLE";return;}
            if(simulation.Step>0)Build();run=Guid.NewGuid().ToString("N");sequence=0;running=true;paused=false;status="CONNECTING";
            bridge=new UnityRustBridge{MaxFrameBytes=8388608};bridge.Start(port);lastReply=Time.realtimeSinceStartup;
        }
        void Update()
        {
            if(running)
            {
                try
                {
                    while(bridge.Incoming.TryDequeue(out JObject reply))
                    {
                        if((string)reply["type"]=="error")throw new IOException((string)reply["reason"]);
                        if((string)reply["protocol_version"]!="2.0"||(string)reply["run_id"]!=run||(long)reply["seq_no"]!=expected)throw new IOException("map response contract mismatch");
                        if(!entranceDeployment&&(bool?)reply["distributed_swarm"]!=true)throw new IOException("この散布設定に対応するモデルが必要です。");
                        waiting=false;lastReply=Time.realtimeSinceStartup;latency=(lastReply-lastSent)*1000;lastMap=reply;entropy=(float)reply["mean_entropy_bits"];map.Update(reply);simulation.ApplyGuidance(reply);evaluation.Evaluate(simulation,reply);status=paused?"PAUSED":"EXPLORING";
                    }
                    if(bridge.Failure!=null)throw new IOException(bridge.Failure);
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
                                    var packet=simulation.Observe(run,sequence++);packet.request_surface=true;string json=JsonConvert.SerializeObject(packet);bridge.SendJson(json);audit.Add(json);expected=packet.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;
                                }
                            }
                        }
                        if(paused&&!waiting&&Time.realtimeSinceStartup-lastSent>5)
                        {
                            var heartbeat=new RescuePacket{run_id=run,seq_no=sequence++,sim_step=simulation.Step,request_surface=true};string json=JsonConvert.SerializeObject(heartbeat);bridge.SendJson(json);audit.Add(json);expected=heartbeat.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;
                        }
                    }
                    if(simulation.Step>=maxSteps)
                    {
                        simulation.Power(false);
                        if(!waiting&&(lastMap==null||(long)lastMap["sim_step"]<maxSteps)){var packet=simulation.Observe(run,sequence++);packet.request_surface=true;string json=JsonConvert.SerializeObject(packet);bridge.SendJson(json);audit.Add(json);expected=packet.seq_no;waiting=true;lastSent=Time.realtimeSinceStartup;}
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
            physicalView.Step(Time.unscaledDeltaTime);inferenceView.Step(Time.unscaledDeltaTime);
            var target=closeup?simulation.Robots[Mathf.Clamp(selectedNode,0,simulation.Robots.Count-1)].Core.position:new Vector3(0,entranceDeployment?.06f:.32f,1.30f);
            var initialOffset=closeup?new Vector3(.085f,.065f,-.065f):new Vector3(2.05f,2f,-.6f)-target;
            physicalView.Apply(physicalCamera,target,initialOffset,closeup?.072f:entranceDeployment?1.12f:1.38f);
            ApplyMapCamera();
            map.SetViewport(MapRenderArea.width/1600f*Screen.width,MapRenderArea.height/900f*Screen.height,1.30f/mapCamera.orthographicSize);
            var robotTarget=simulation.Robots[Mathf.Clamp(selectedNode,0,simulation.Robots.Count-1)].Core.position;robotCamera.enabled=!closeup;robotCamera.transform.position=robotTarget+new Vector3(.060f,.055f,-.065f);robotCamera.transform.LookAt(robotTarget);
            for(int i=0;i<simulation.Robots.Count;i++)foreach(var renderer in simulation.Robots[i].Root.GetComponentsInChildren<Renderer>())renderer.gameObject.layer=i==selectedNode?9:0;
            if(filmView!="")
            {
                bool mapping=filmView=="map",macro=filmView=="robot";
                robotCamera.enabled=false;physicalCamera.enabled=!mapping;mapCamera.enabled=mapping;
                var camera=mapping?mapCamera:physicalCamera;camera.rect=new Rect(0,0,1,1);camera.backgroundColor=new Color(.018f,.029f,.039f);
                float t=Time.realtimeSinceStartup;var focus=macro?robotTarget:new Vector3(0,.15f,1.32f);
                var offset=macro?new Vector3(.062f,.044f,-.057f):mapping?new Vector3(2.35f,1.95f,-2.4f):new Vector3(2.3f,1.75f,-1.6f);
                offset=Quaternion.AngleAxis(Mathf.Sin(t*.024f)*12,Vector3.up)*offset;camera.transform.position=focus+offset;camera.transform.LookAt(focus);
                camera.orthographicSize=macro?.043f:mapping?1.12f:1.08f;physicalCamera.cullingMask=macro?1<<9:~(1<<8);
                map.SetViewport(Screen.width,Screen.height,1.30f/mapCamera.orthographicSize);
            }
        }
        void OnApplicationFocus(bool focused){if(!focused){physicalView.Cancel();inferenceView.Cancel();}}
        void Export()
        {
            if(audit.Count==0)return;string path=Path.Combine(Application.persistentDataPath,"rescue-runs");Directory.CreateDirectory(path);var output=Path.Combine(path,run+".jsonl");File.WriteAllLines(output,audit);if(lastMap!=null)File.AppendAllText(output,lastMap.ToString(Formatting.None)+Environment.NewLine);File.WriteAllText(Path.Combine(path,run+"-evaluation.json"),JsonConvert.SerializeObject(evaluation.Snapshot(simulation),Formatting.Indented));
        }
        void OnDestroy(){if(record&&frameFolder!=null)File.WriteAllLines(Path.Combine(frameFolder,"timestamps.csv"),frameTimes);bridge?.Dispose();bundled?.Dispose();simulation?.Dispose();map?.Dispose();}
        void Fill(Rect r,Color c){GUI.color=c;GUI.DrawTexture(r,Texture2D.whiteTexture);GUI.color=Color.white;}
        void Text(float x,float y,float w,float h,string value,int size,Color color,bool bold=false)
        {
            var style=new GUIStyle(GUI.skin.label){font=font,fontSize=size,fontStyle=bold?FontStyle.Bold:FontStyle.Normal,wordWrap=true};style.normal.textColor=color;GUI.Label(new Rect(x,y,w,h),value,style);
        }
        void Rounded(Rect area,Color color)
        {
            GUI.DrawTexture(area,Texture2D.whiteTexture,ScaleMode.StretchToFill,true,0,color,0,7);
        }
        bool Button(float x,float y,float width,string label,bool selected=false,float height=30,bool primary=false)
        {
            var rect=new Rect(x,y,width,height);var bg=primary?Mint:selected?new Color(.13f,.29f,.27f):new Color(.105f,.145f,.17f);
            if(rect.Contains(Event.current.mousePosition)&&GUI.enabled)bg=Color.Lerp(bg,Color.white,.07f);Rounded(rect,bg);
            var style=new GUIStyle(GUI.skin.button){font=font,fontSize=primary?15:12,fontStyle=primary?FontStyle.Bold:FontStyle.Normal};style.normal.background=null;style.hover.background=null;style.active.background=null;style.focused.background=null;
            style.normal.textColor=primary?Background:selected?Mint:Ink;style.hover.textColor=style.normal.textColor;style.active.textColor=style.normal.textColor;
            return GUI.Button(rect,label,style);
        }
        void Header(float x,float y,float width,string number,string title,string note)
        {
            Fill(new Rect(x,y,width,38),Panel);Text(x+12,y+8,28,28,number,12,Mint,true);Text(x+46,y+6,width-100,28,title,18,Ink,true);Text(x+12,y+452,width-24,30,note,12,Muted);
        }
        Rect UtilityBounds=>new Rect(utilityMenu=="view"?1030:1164,122,260,utilityMenu=="view"?100:100);
        void UtilityOverlay()
        {
            if(utilityMenu=="")return;var area=UtilityBounds;Rounded(area,new Color(.055f,.08f,.105f,.99f));
            if(utilityMenu=="view")
            {
                if(Button(area.x+10,area.y+12,240,"正解位置の赤球 "+(showTruth?"ON":"OFF"),showTruth,32)){showTruth=!showTruth;simulation.Scene.ShowTruth(showTruth);utilityMenu="";}
                if(Button(area.x+10,area.y+54,240,"両画面の視点を戻す",false,32)){physicalView.Reset();inferenceView.Reset();utilityMenu="";}
            }
            else
            {
                if(Button(area.x+10,area.y+12,240,"自動接続 "+(simulation.Docking?"ON":"OFF"),simulation.Docking,32)){simulation.Docking=!simulation.Docking;if(!simulation.Docking)simulation.ReleaseAll();utilityMenu="";}
                if(Button(area.x+10,area.y+54,240,"すべての接続を解除",false,32)){simulation.ReleaseAll();utilityMenu="";}
            }
        }
        void Badge(float x,float y,float width,string label,Color color)
        {Rounded(new Rect(x,y,width,27),new Color(.04f,.065f,.08f,.92f));Fill(new Rect(x+8,y+9,3,9),color);Text(x+18,y+5,width-22,22,label,12,Color.Lerp(color,Color.white,.25f),true);}
        void TruthOverlay()
        {
            if(!showTruth||closeup)return;
            int index=0;foreach(var position in simulation.Scene.VictimPositions)
            {
                index++;var p=physicalCamera.WorldToViewportPoint(position);if(p.z<=0||p.x<.02f||p.x>.93f||p.y<.06f||p.y>.94f)continue;
                float px=WorldArea.x+p.x*WorldArea.width,py=WorldArea.y+(1-p.y)*WorldArea.height;
                Rounded(new Rect(px+8,py-14,122,24),new Color(.04f,.06f,.07f,.90f));Text(px+14,py-11,112,22,$"要救助者 {index} / 正解",11,new Color(1,.35f,.38f),true);
            }
        }
        void OnGUI()
        {
            if(filmView!="")return;
            if(simulation==null)return;GUI.matrix=Matrix4x4.Scale(new Vector3(Screen.width/1600f,Screen.height/900f,1));
            // Automated frame capture must not drift when unrelated mouse input arrives.
            if(record&&Event.current.type!=EventType.Repaint&&Event.current.type!=EventType.Layout)Event.current.Use();
            if(utilityMenu!=""&&Event.current.type==EventType.MouseDown&&Event.current.mousePosition.y>122&&!UtilityBounds.Contains(Event.current.mousePosition)){utilityMenu="";Event.current.Use();}
            physicalView.Input(WorldArea,scenePicker||utilityMenu!="",4121);inferenceView.Input(MapRenderArea,scenePicker||utilityMenu!="",4122);
            Fill(new Rect(0,0,1600,122),Background);Fill(new Rect(0,572,1600,328),Background);Fill(new Rect(0,122,24,450),Background);Fill(new Rect(794,122,16,450),Background);Fill(new Rect(1576,122,24,450),Background);
            var spec=RescueScene.Catalog[sceneIndex];
            TruthOverlay();
            Text(24,13,450,23,"PURUMOGU   /   FIELD SYSTEM",12,Mint,true);Text(24,39,460,38,"災害空隙サーチ",27,Ink,true);
            Text(510,16,490,22,"LOCAL VOID  ·  07 CM  ·  SENSOR FUSION",11,Muted);Text(510,43,490,26,spec.Disaster+" / "+spec.Feature,13,Ink);
            Rounded(new Rect(1030,22,100,38),Panel);Text(1041,32,84,24,running?(paused?"一時停止中":"探索中"):failure!=""?"停止":"準備完了",12,running?Mint:Muted,true);
            if(Button(1142,22,82,$"{speed:F0}×",false,38))speed=speed<2?4:1;
            if(Button(1238,22,198,running?(paused?"探索を再開":"一時停止"):"散布・探索を開始",false,38,true)){if(running)paused=!paused;else Begin();}
            if(Button(1448,22,128,"リセット",false,38))Build();
            if(Button(24,86,430,$"{spec.Id}   {spec.Name}   /   {new[]{"易","中","難"}[spec.Difficulty]}  ▾",scenePicker)){if(!scenePicker)sceneScroll=new Vector2(0,sceneIndex*27);scenePicker=!scenePicker;}
            bool previousEnabled=GUI.enabled;GUI.enabled=!running;
            Text(480,94,38,22,"台数",12,Muted);
            for(int i=0;i<3;i++){int count=new[]{24,36,48}[i];if(Button(520+i*56,86,50,count.ToString(),nodeCount==count)){nodeCount=count;entranceDeployment=false;Build();}}
            if(Button(700,86,134,"配置を変える")){deploymentSeed++;sourceSeed=Guid.NewGuid().GetHashCode();entranceDeployment=false;Build();}
            GUI.enabled=previousEnabled;Text(848,94,172,22,running?"設定変更はリセット後":"現場全域に分散投下",11,Muted);
            if(Button(1030,86,122,"表示設定  ▾",utilityMenu=="view"))utilityMenu=utilityMenu=="view"?"":"view";
            if(Button(1164,86,160,$"接続 {simulation.Links.Count}  /  {(simulation.Docking?"自動":"手動")}  ▾",utilityMenu=="dock"))utilityMenu=utilityMenu=="dock"?"":"dock";
            if(Button(1336,86,112,"記録を保存"))Export();Text(1464,94,112,22,$"{maxSteps*.02f:F0} s / CPU",11,Muted);
            Header(24,126,770,"01","ロボットと災害現場",$"{status}  ·  {simulation.Step*.02f:F1} s  ·  接続 {simulation.Links.Count} / 累計 {simulation.Attached}  ·  解除 {simulation.Detached}  ·  透視中も床板に接地");
            Text(365,140,212,20,"ドラッグ回転 / ホイール拡縮",10,Muted);
            if(Button(584,132,60,"機体",closeup,25)){closeup=!closeup;physicalView.ResetZoom();}
            if(Button(650,132,60,"透視",cutaway,25)){cutaway=!cutaway;simulation.Scene.Cutaway(cutaway);}
            if(Button(716,132,66,"戻す",false,25))physicalView.Reset();
            Fill(new Rect(24,620,770,256),Panel);Fill(new Rect(810,620,766,256),Panel);
            Text(38,630,730,26,$"03  各ノードの状態 / 投下 {simulation.Robots.FindAll(r=>r.Released).Count} / {simulation.Robots.Count} 台",18,Ink,true);Text(42,666,730,22,"NODE    電池      関節 / 回転     接続    受信peer    RSSI     音響SNR    Δ温度    状態",12,Muted,true);
            nodeScroll=GUI.BeginScrollView(new Rect(36,690,744,178),nodeScroll,new Rect(0,0,722,Mathf.Max(178,simulation.Robots.Count*17)),false,true);
            for(int i=0;i<simulation.Robots.Count;i++)
            {
                var node=simulation.Robots[i];float y=i*17;if(i==selectedNode)Fill(new Rect(0,y,722,17),new Color(.10f,.20f,.22f));
                string row=$"{node.Id}     {node.Energy*100,3:F0}%     {node.BendDegrees,3:F0}°/{node.SpinDegrees,3:F0}°     {node.ConnectionCount}        {node.Peers,2}          {node.Signal,4:F0}       {node.LastSnr,4:F1}       {node.Temperature,4:F1}°     {(!node.Released?"投下待機":node.Lost?"喪失":node.Verifying?"確認中":node.Stuck?"停滞":node.Powered?"稼働":"停止")}";
                Text(6,y,710,20,row,11,node.Lost||node.Stuck?Amber:Ink);if(GUI.Button(new Rect(0,y,722,17),GUIContent.none,GUIStyle.none))selectedNode=i;
            }
            GUI.EndScrollView();
            var sensorNode=simulation.Robots[Mathf.Clamp(selectedNode,0,simulation.Robots.Count-1)];Text(548,636,232,23,$"{sensorNode.Id}  ガス {sensorNode.GasLevel:F2} / 水深 {sensorNode.WaterDepth*100:F1}cm",11,Muted);
            Text(826,630,730,26,"04  推論の評価 / 真値はこの区画だけで使用",18,Ink,true);
            float humanPrecision=evaluation.CueCount>0?(float)evaluation.Matched/evaluation.CueCount:0,humanRecall=simulation.Scene.Spec.Victims>0?(float)evaluation.Matched/simulation.Scene.Spec.Victims:0;
            Metric(832,670,"3D占有IoU / 全領域",$"{evaluation.GlobalIoU*100:F1}%",evaluation.GlobalIoU>=.95f);Metric(1070,670,"要救助者候補の適合率",evaluation.CueCount>0?$"{humanPrecision*100:F1}%":"候補なし",humanPrecision>=.95f);Metric(1320,670,"要救助者候補の再現率",simulation.Scene.Spec.Victims>0?$"{humanRecall*100:F1}%":"負例",humanRecall>=.95f);
            Text(834,745,700,23,$"全推定地図を評価（右上は観測の進行表示） / 観測率 {evaluation.Coverage*100:F1}% · 既観測IoU {evaluation.IoU:F2} · 位置RMSE {evaluation.PoseRmse*100:F1} cm",13,Ink);
            Text(834,777,700,24,$"候補 {evaluation.CueCount} / 一致 {evaluation.Matched} / 誤警報 {evaluation.FalseAlarms} · 位置誤差 {(float.IsNaN(evaluation.LocalizationError)?"未評価":(evaluation.LocalizationError*100).ToString("F1")+" cm")} · Rust {latency:F0} ms",13,Muted);
            bool bootstrap=lastMap!=null&&(bool?)lastMap["person_classifier_ready"]==false;
            string context=bootstrap?"格子：人候補は橙 / 浸水は青 / ガスは赤 / 推定補完は暗い破線。新形状の精度は検証中です。":"地図・人候補と環境反応を区別して表示します。ガス反応は実測濃度や確定した漏洩位置を表しません。";
            Text(834,810,700,48,string.IsNullOrEmpty(failure)?context:failure,12,string.IsNullOrEmpty(failure)?Muted:Amber);
            UtilityOverlay();
            if(scenePicker)
            {
                Fill(new Rect(24,125,750,460),new Color(.045f,.075f,.09f,.99f));Text(40,138,710,28,"30 SCENES / 多形状・複数階・立体分岐",18,Mint,true);
                sceneScroll=GUI.BeginScrollView(new Rect(36,173,732,396),sceneScroll,new Rect(0,0,710,RescueScene.Catalog.Length*27),false,true);
                for(int i=0;i<RescueScene.Catalog.Length;i++){var item=RescueScene.Catalog[i];if(Button(4,i*27,698,$"{item.Id}  {item.Name}  /  {item.Disaster}  ·  {item.Feature}",i==sceneIndex,24)){sceneIndex=i;scenePicker=false;Build();}}
                GUI.EndScrollView();
            }
        }
        void Metric(float x,float y,string label,string value,bool passed){Text(x,y,225,24,label,12,Muted);Text(x,y+24,225,42,value,30,passed?Mint:Amber,true);}
    }
}
