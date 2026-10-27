using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Purumogu {
    public sealed class AppController:MonoBehaviour {
        enum ScreenState{Prepare,Explore,Result,Recover}
        ScreenState screen;
        SimulationWorld world;UnityRustBridge bridge;
        Camera overview,detail;Font font;
        string run,reason="Waiting for the first observation.",status="READY TO CONNECT",mode="CLUSTER",entry="--",policy="ghost_flow";
        readonly float[] beliefs={.19f,.19f,.19f,.19f,.19f,.05f};
        readonly Queue<string> recent=new Queue<string>();
        readonly List<string> export=new List<string>();
        string modelVersion="--";float entropy=2.46f,eig;
        bool hello,observing,abstain=true,waitingAck;int family=4,seed=2027261,batch,batchSteps,port=47110,impactCount,radioCount,coupleCount;
        long sequence;float accumulator,lastReply,delay;ActionCommand active,next;
        bool record,macroView;int frame,sensorSeed=45267;string frames;float captureTime;
        readonly List<string> frameTimes=new List<string>{"frame_index,wall_time_s,sim_step"};
        static readonly Color Ink=new Color(.08f,.13f,.17f),Muted=new Color(.44f,.53f,.56f),Mint=new Color(.22f,.81f,.66f),Amber=new Color(.98f,.66f,.28f);
        void Start(){
            Application.runInBackground=true;Application.targetFrameRate=60;
            font=Font.CreateDynamicFontFromOSFont(new[]{"Yu Gothic","Segoe UI","Arial"},20);
            var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++){if(args[i]=="--port")int.TryParse(args[i+1],out port);if(args[i]=="--family")int.TryParse(args[i+1],out family);if(args[i]=="--map-seed")int.TryParse(args[i+1],out seed);if(args[i]=="--sensor-seed")int.TryParse(args[i+1],out sensorSeed);if(args[i]=="--view")macroView=args[i+1]=="macro";if(args[i]=="--frames"){record=true;frames=args[i+1];Directory.CreateDirectory(frames);}}
            BuildWorld();if(record)Begin();
        }
        void BuildWorld(){world?.Dispose();world=new SimulationWorld();world.Build(family,seed,seed+10000,sensorSeed);if(!overview){var cameraObject=new GameObject("OverviewCamera");overview=cameraObject.AddComponent<Camera>();overview.clearFlags=CameraClearFlags.SolidColor;overview.backgroundColor=new Color(.90f,.93f,.92f);overview.orthographic=true;overview.orthographicSize=.93f;overview.nearClipPlane=.01f;overview.farClipPlane=20;
                detail=new GameObject("NodeDetailCamera").AddComponent<Camera>();detail.clearFlags=CameraClearFlags.SolidColor;detail.backgroundColor=new Color(.13f,.19f,.22f);detail.depth=1;detail.orthographic=true;detail.orthographicSize=.073f;detail.nearClipPlane=.002f;detail.farClipPlane=4;detail.rect=new Rect(.466f,.52f,.175f,.205f);detail.enabled=false;
                var light=new GameObject("SoftKey").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.2f;light.transform.rotation=Quaternion.Euler(45,-30,0);light.shadows=LightShadows.Soft;RenderSettings.ambientMode=UnityEngine.Rendering.AmbientMode.Flat;RenderSettings.ambientLight=new Color(.65f,.72f,.75f);}
            overview.rect=new Rect(0,122f/900f,.67f,666f/900f);overview.transform.position=new Vector3(.82f,1.75f,-1.0f);overview.transform.LookAt(new Vector3(0,-.05f,0));
            if(macroView)detail.rect=overview.rect;
        }
        void Begin(){bridge?.Dispose();run=Guid.NewGuid().ToString("N");sequence=0;hello=false;batch=0;observing=false;abstain=true;waitingAck=false;export.Clear();bridge=new UnityRustBridge();bridge.Start(port);status="CONNECTING";screen=ScreenState.Explore;lastReply=Time.realtimeSinceStartup;}
        ObservationPacket Base(string type)=>new ObservationPacket{type=type,run_id=run,seq_no=sequence++,sim_step=world.Step,timestamp_ms=world.Step*20L,remaining_nodes=20-world.Deployed};
        void Send(ObservationPacket p){bridge.Send(p);if(export.Count>=10000)throw new IOException("audit capacity exceeded");export.Add(Wire.Serialize(p));}
        void Update(){
            if(bridge!=null&&screen==ScreenState.Explore){try{
                if(bridge.Failure!=null){Stop("SAFE STOP / "+bridge.Failure);return;}
                if(bridge.Connected&&!hello){var p=Base("hello");p.policy=policy;p.policy_seed=20261027;Send(p);hello=true;status="HANDSHAKE";}
                while(bridge.Incoming.TryDequeue(out JObject msg)){lastReply=Time.realtimeSinceStartup;Handle(msg);}
                if(hello&&Time.realtimeSinceStartup-lastReply>12){Stop("SAFE STOP / Rust response timed out");return;}
                if(next!=null&&!observing&&Time.realtimeSinceStartup>=delay){active=next;next=null;bool accepted=world.Deploy(active,out string error);var ack=Base("action_ack");ack.action_id=active.action_id;ack.accepted=accepted;Send(ack);if(!accepted){Stop(error);return;}mode=active.formation_mode;entry="E"+active.entry_id;waitingAck=true;status="ACTUATOR / WAITING FOR ACK";}
                if(observing){accumulator+=Time.deltaTime;int steps=0;while(accumulator>=SimulationWorld.Dt&&steps<8){accumulator-=SimulationWorld.Dt;world.Tick();batchSteps++;steps++;if(batchSteps%10==0)Flush(false);if(batchSteps>=450){Flush(true);observing=false;batch++;status="RUST / ESTIMATING";break;}}}
            }catch(Exception e){Stop("SAFE STOP / "+e.Message);}}
            if(detail){detail.enabled=screen!=ScreenState.Prepare&&world.Nodes.Count>0;if(detail.enabled){var node=world.Nodes[world.Nodes.Count-1];Vector3 target=node.Body.position;detail.transform.position=target+world.Terrain.Up*.25f+world.Terrain.Forward*.01f;detail.transform.LookAt(target,world.Terrain.Forward);}}
            if(record&&Time.realtimeSinceStartup>=captureTime){captureTime=Time.realtimeSinceStartup+.05f;frameTimes.Add(frame+","+Time.realtimeSinceStartup.ToString("F5",CultureInfo.InvariantCulture)+","+world.Step);ScreenCapture.CaptureScreenshot(Path.Combine(frames,$"frame-{frame++:D5}.png"));if(frame>=760)Application.Quit();}
        }
        void Handle(JObject m){if((string)m["protocol_version"]!="1.0"||(string)m["run_id"]!=run)throw new IOException("reply contract mismatch");modelVersion=(string)m["model_version"]??modelVersion;string type=(string)m["type"];
            if(type=="action"){if(next!=null)throw new IOException("overlapping action");next=m.ToObject<ActionCommand>();eig=(float?)m["expected_information_gain_bits"]??0;delay=Time.realtimeSinceStartup+1.5f;status="PARENT / NEXT COMMAND";}
            else if(type=="ack"&&waitingAck){waitingAck=false;observing=true;batchSteps=0;accumulator=0;status="EXPLORING";}
            else if(type=="inference"){abstain=(bool)m["abstain"];reason=(string)m["reason"];entropy=(float)m["entropy_bits"];var list=(JArray)m["beliefs"];for(int i=0;i<Math.Min(list.Count,6);i++)beliefs[i]=(float)list[i]["probability"];}
            else if(type=="result"){screen=ScreenState.Result;status="BUDGET COMPLETE";world.SafeStop();Export();}
            else if(type=="error")Stop((string)m["reason"]);
        }
        void Flush(bool end){while(world.Pending.Count>Wire.MaxEvents)FlushChunk(Wire.MaxEvents,false);if(world.Pending.Count>0||end)FlushChunk(world.Pending.Count,end);}
        void FlushChunk(int n,bool end){var p=Base("observation_batch");p.events=world.Pending.GetRange(0,n);world.Pending.RemoveRange(0,n);p.batch_end=end;p.action_id=active.action_id;p.entry_id=active.entry_id;p.formation_mode=active.formation_mode;p.node_count=active.node_count;
            foreach(var e in p.events){if(e.kind=="IMPACT")impactCount++;if(e.kind=="PEER_HEARD")radioCount++;if(e.kind=="COUPLE_START")coupleCount++;if(e.kind!="PEER_HEARD"&&e.kind!="OBSERVATION_GAP"){recent.Enqueue($"{e.t_ms/1000f,5:F1}s   {e.node_id,-5} {e.kind}");while(recent.Count>7)recent.Dequeue();}}Send(p);}
        void Stop(string why){observing=false;next=null;waitingAck=false;world.SafeStop();status=why;reason=why;abstain=true;bridge?.Dispose();bridge=null;screen=ScreenState.Result;Export();}
        void Export(){if(export.Count==0)return;string dir=Path.Combine(Application.persistentDataPath,"runs");Directory.CreateDirectory(dir);File.WriteAllLines(Path.Combine(dir,run+".jsonl"),export);}
        void OnDestroy(){if(record)File.WriteAllLines(Path.Combine(frames,"timestamps.csv"),frameTimes);bridge?.Dispose();world?.Dispose();}
        void Rect(float x,float y,float w,float h,Color color){GUI.color=color;GUI.DrawTexture(new Rect(x,y,w,h),Texture2D.whiteTexture);GUI.color=Color.white;}
        void Text(float x,float y,float w,float h,string value,int size,Color color,FontStyle style=FontStyle.Normal){var s=new GUIStyle(GUI.skin.label){font=font,fontSize=size,fontStyle=style,wordWrap=true};s.normal.textColor=color;GUI.Label(new Rect(x,y,w,h),value,s);}
        bool Button(float x,float y,float w,string label,Color color){Rect(x,y,w,46,color);var s=new GUIStyle(GUI.skin.button){font=font,fontSize=17};s.normal.background=null;s.hover.background=null;s.active.background=null;s.normal.textColor=Ink;return GUI.Button(new Rect(x,y,w,46),label,s);}
        void OnGUI(){if(world==null)return;GUI.matrix=Matrix4x4.Scale(new Vector3(Screen.width/1600f,Screen.height/900f,1));
            Rect(0,0,1600,112,new Color(.95f,.97f,.95f));Text(36,21,660,42,"PURUMOGU  /  ぷるもぐサーチ",29,Ink,FontStyle.Bold);Text(38,67,850,30,"A physical sensing experiment  ·  Unity × Rust  ·  simulation only",16,Muted);
            string[] names={"01  Prepare","02  Explore","03  Result","04  Recover"};for(int i=0;i<4;i++){bool selected=(int)screen==i;Text(970+i*150,42,150,36,names[i],16,selected?Ink:Muted,selected?FontStyle.Bold:FontStyle.Normal);if(selected)Rect(970+i*150,83,124,3,Mint);}
            Rect(1072,112,528,788,new Color(.97f,.98f,.96f));Rect(1100,145,460,1,new Color(.81f,.85f,.83f));
            Text(1110,162,420,24,"RUST / CANDIDATE FAMILIES",14,Muted,FontStyle.Bold);Text(1110,198,420,45,abstain?"Still uncertain":"Provisional estimate",30,abstain?new Color(.70f,.39f,.12f):Ink,FontStyle.Bold);
            Text(1110,245,420,65,"These bars compare graph families. They do not reconstruct a 3D map.",17,Muted);
            for(int i=0;i<6;i++){float y=322+i*56;Text(1110,y,290,25,ScenarioBuilder.Families[i],16,Ink);Text(1452,y,80,25,$"{beliefs[i]*100:F1}%",16,Ink);Rect(1110,y+29,420,7,new Color(.85f,.89f,.86f));Rect(1110,y+29,420*beliefs[i],7,i==5?Amber:Mint);}
            Text(1110,678,420,35,$"Entropy  {entropy:F2} bits",21,Ink);Text(1110,718,420,75,reason,16,Muted);Text(1110,830,420,35,"Model  "+modelVersion,13,Muted);
            Rect(0,778,1072,122,new Color(.06f,.12f,.15f));Text(38,796,1000,25,status,16,Mint,FontStyle.Bold);Text(38,836,300,40,$"{world.Step*.02f:F1}s  /  50 Hz",25,Color.white);Text(355,838,300,34,$"{world.Deployed} / 20 nodes",22,Color.white);Text(685,838,350,34,$"{batch} / 3 batches",22,Color.white);
            if(screen==ScreenState.Prepare){Rect(30,130,1000,175,new Color(.94f,.97f,.94f));Text(52,145,950,32,"Choose a simulation. Let the observations do the estimating.",23,Ink,FontStyle.Bold);for(int i=0;i<5;i++)if(Button(52+i*187,198,176,ScenarioBuilder.Families[i],family==i?Mint:new Color(.84f,.89f,.86f))){family=i;BuildWorld();}Text(52,257,920,28,"20 available nodes · 3 gates · CLUSTER / CHAIN_INSERT · CPU inference",16,Muted);
                if(Button(52,700,200,"Start exploration",Mint))Begin();if(Button(274,700,190,"Policy: "+policy,new Color(.85f,.89f,.86f)))policy=policy=="ghost_flow"?"random":policy=="random"?"fixed":policy=="fixed"?"passive":"ghost_flow";
            }else{
                Rect(30,130,510,78,new Color(.94f,.97f,.94f));Text(48,141,470,24,"PARENT / "+mode,17,Ink,FontStyle.Bold);Text(48,172,480,24,$"Gate {entry} · Expected gain {eig:F3} bits · {policy}",15,Muted);
                Text(36,220,760,24,"SIMULATION CUTAWAY / observer view; geometry is excluded from AI input",13,Muted);
                if(world.Nodes.Count>0){Rect(746,432,280,26,new Color(.94f,.97f,.94f));Text(750,432,275,26,"LOCAL NODE / capture arms + body",12,Muted);}
                Rect(38,590,470,165,new Color(.94f,.97f,.94f));Text(54,600,440,26,$"EVENTS   impact {impactCount}   radio {radioCount}   coupled {coupleCount}",14,Ink,FontStyle.Bold);int line=0;foreach(var item in recent)Text(54,630+line++*16,440,20,item,12,Muted);
                if(screen==ScreenState.Explore){if(Button(830,700,195,"Stop safely",Amber))Stop("STOPPED BY OPERATOR");}
                if(screen==ScreenState.Result){if(Button(585,640,195,"Export log",Mint))Export();if(Button(805,640,220,"Go to recovery",new Color(.84f,.89f,.86f)))screen=ScreenState.Recover;Text(585,700,445,50,"Compare policies with scripts/Run-Experiments.ps1. The results are measured on held-out maps.",15,Muted);}
                if(screen==ScreenState.Recover){Text(585,590,440,50,$"Ready to recover {world.Nodes.Count} simulated nodes.",21,Ink);if(Button(585,680,240,"Recover and reset",Mint)){bridge?.Dispose();bridge=null;BuildWorld();screen=ScreenState.Prepare;impactCount=radioCount=coupleCount=0;recent.Clear();}Text(585,735,440,28,"Stage cleanup does not demonstrate real-world retrieval.",13,Muted);}
            }
        }
    }
}
