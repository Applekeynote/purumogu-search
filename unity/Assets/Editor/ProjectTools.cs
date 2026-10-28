using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug=UnityEngine.Debug;

namespace Purumogu.Editor {
    public static class ProjectTools {
        static string Arg(string key,string fallback){var a=Environment.GetCommandLineArgs();for(int i=0;i<a.Length-1;i++)if(a[i]==key)return a[i+1];return fallback;}
        [MenuItem("Purumogu/Create main scene")]
        public static void Initialize(){
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            var go=new GameObject("PurumoguApplication");go.AddComponent<Purumogu.Rescue.RescueController>();
            EditorSceneManager.SaveScene(SceneManager.GetActiveScene(),"Assets/Scenes/Main.unity");
            EditorBuildSettings.scenes=new[]{new EditorBuildSettingsScene("Assets/Scenes/Main.unity",true)};
            PlayerSettings.companyName="Purumogu Research";PlayerSettings.productName="Purumogu Search";PlayerSettings.defaultScreenWidth=1600;PlayerSettings.defaultScreenHeight=900;PlayerSettings.runInBackground=true;
            PlayerSettings.colorSpace=ColorSpace.Linear;
            Directory.CreateDirectory("Assets/Resources");
            foreach(string shader in new[]{"Standard","Sprites/Default"}){string asset="Assets/Resources/"+shader.Replace('/','_')+".mat";if(!File.Exists(asset))AssetDatabase.CreateAsset(new Material(Shader.Find(shader)),asset);}
            AssetDatabase.SaveAssets();Debug.Log("PURUMOGU_INITIALIZED");
        }
        public static void Run(){try{string task=Arg("-purumoguTask","init");if(task=="init")Initialize();else if(task=="collect")Collect();else if(task=="compare")Compare();else if(task=="smoke")Smoke();else if(task=="rescue-smoke")RescueTests.Smoke();else if(task=="rescue-docking-smoke")RescueTests.DockingSmoke();else if(task=="rescue-swarm-smoke")RescueTests.DistributedSmoke();else if(task=="rescue-swarm-evaluate")RescueTests.EvaluateSwarm();else if(task=="rescue-audit")RescueTests.AuditSources();else if(task=="rescue-evaluate")RescueTests.Evaluate();else if(task=="build")Build();else throw new ArgumentException("unknown task");EditorApplication.Exit(0);}catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}}
        static string Output=>Arg("-purumoguOutput",Path.GetFullPath(Path.Combine(Application.dataPath,"../../datasets/generated")));
        static void ResetScene(){EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);}
        static ObservationPacket Packet(SimulationWorld w,string run,ActionCommand a,ref long sequence,List<SensorEvent> events,bool end){return new ObservationPacket{run_id=run,seq_no=++sequence,sim_step=w.Step,timestamp_ms=w.Step*20L,events=events,batch_end=end,action_id=a.action_id,entry_id=a.entry_id,formation_mode=a.formation_mode,node_count=a.node_count,remaining_nodes=20-w.Deployed};}
        static void Drain(SimulationWorld w,string run,ActionCommand a,ref long sequence,bool end,Action<ObservationPacket> send){
            while(w.Pending.Count>Wire.MaxEvents){var chunk=w.Pending.GetRange(0,Wire.MaxEvents);w.Pending.RemoveRange(0,Wire.MaxEvents);send(Packet(w,run,a,ref sequence,chunk,false));}
            if(w.Pending.Count>0||end){var chunk=new List<SensorEvent>(w.Pending);w.Pending.Clear();send(Packet(w,run,a,ref sequence,chunk,end));}
        }
        static object Truth(SimulationWorld w,string run,int family,int mapIndex,int seed,int sensor,string split,string policy,double loss=0)=>new{run_id=run,map_id="map-"+family+"-"+mapIndex,label=family,split,policy,map_seed=seed,physics_seed=seed+10000,sensor_seed=sensor,max_depth_m=w.MaxDepth,passed_nodes=w.PassedNodes,packet_loss=loss};
        static string Partition(int i)=>i<10?"train":i<13?"valid":"test";
        static void Collect(){ResetScene();Directory.CreateDirectory(Output);int maps=int.Parse(Arg("-purumoguMaps","18"));int steps=int.Parse(Arg("-purumoguSteps","450"));
            using(var obs=new StreamWriter(Path.Combine(Output,"observations.jsonl"),false,new UTF8Encoding(false)))using(var truth=new StreamWriter(Path.Combine(Output,"truth.jsonl"),false,new UTF8Encoding(false))){
                var watch=Stopwatch.StartNew();int episodes=0;
                for(int family=0;family<5;family++)for(int map=0;map<maps;map++)for(int initial=0;initial<6;initial++){
                    int seed=2026000+map*97,sensor=42100+map*19;string run=Guid.NewGuid().ToString("N");long seq=0;
                    using(var world=new SimulationWorld()){world.Build(family,seed,seed+10000,sensor);
                        for(int batch=0;batch<3;batch++){int a=(initial+batch*2)%6;var command=new ActionCommand{action_id="a-"+batch.ToString("D3"),entry_id=a/2,node_count=5,formation_mode=a%2==0?"CLUSTER":"CHAIN_INSERT",policy="fixed"};if(!world.Deploy(command,out string reason))throw new Exception(reason);
                            for(int step=0;step<steps;step++){world.Tick();if(step%10==9)Drain(world,run,command,ref seq,false,p=>obs.WriteLine(Wire.Serialize(p)));}
                            Drain(world,run,command,ref seq,true,p=>obs.WriteLine(Wire.Serialize(p)));
                        }
                        truth.WriteLine(JsonConvert.SerializeObject(Truth(world,run,family,map,seed,sensor,Partition(map),"balanced")));}
                    episodes++;if(episodes%6==0){obs.Flush();truth.Flush();Debug.Log($"COLLECT family={family} map={map} episodes={episodes} elapsed={watch.Elapsed.TotalSeconds:F1}s");}
                }
                File.WriteAllText(Path.Combine(Output,"collection.json"),JsonConvert.SerializeObject(new{episodes,maps,steps,unity=Application.unityVersion,elapsed_seconds=watch.Elapsed.TotalSeconds,physics_dt=.02,seed_scheme="map=2026000+index*97; physics=map+10000; sensor=42100+index*19"},Formatting.Indented));
            }
        }
        sealed class Client:IDisposable {
            TcpClient socket;StreamReader reader;StreamWriter writer;
            public Client(int port){socket=new TcpClient("127.0.0.1",port);socket.ReceiveTimeout=15000;socket.SendTimeout=15000;socket.NoDelay=true;reader=new StreamReader(socket.GetStream(),Encoding.UTF8);writer=new StreamWriter(socket.GetStream(),new UTF8Encoding(false)){AutoFlush=true};}
            public void Send(ObservationPacket p)=>writer.WriteLine(Wire.Serialize(p));
            public JObject Receive(){string line=reader.ReadLine();if(line==null||line.Length>Wire.MaxLine)throw new IOException("invalid server frame");var v=JObject.Parse(line);if((string)v["type"]=="error")throw new IOException(v.ToString());return v;}
            public void Dispose(){socket.Dispose();}
        }
        static void Compare(){ResetScene();Directory.CreateDirectory(Output);int steps=int.Parse(Arg("-purumoguSteps","450"));int repeats=int.Parse(Arg("-purumoguRepeats","2"));int maps=int.Parse(Arg("-purumoguTestMaps","10"));int port=int.Parse(Arg("-purumoguPort","47110"));bool faults=Arg("-purumoguFaults","false")=="true";
            string stem=faults?"fault":"comparison";
            using(var obs=new StreamWriter(Path.Combine(Output,stem+".jsonl"),false,new UTF8Encoding(false)))using(var truth=new StreamWriter(Path.Combine(Output,stem+"-truth.jsonl"),false,new UTF8Encoding(false))){int count=0;var watch=Stopwatch.StartNew();
                foreach(int family in faults?new[]{0,1,2,3,4,5}:new[]{0,1,2,3,4})for(int map=13;map<13+maps;map++)for(int repeat=0;repeat<repeats;repeat++)foreach(string policy in faults?new[]{"ghost_flow"}:new[]{"ghost_flow","random","fixed","passive"})foreach(double loss in faults?new[]{0.0,.1,.2,.4}:new[]{0.0}){
                    int seed=2026000+map*97,sensor=42100+map*19+repeat*1009;string run=Guid.NewGuid().ToString("N");long seq=0;
                    using(var client=new Client(port))using(var world=new SimulationWorld()){
                        world.Build(family,seed,seed+10000,sensor,loss);client.Send(new ObservationPacket{type="hello",run_id=run,policy=policy,policy_seed=9001+map*37+repeat*101,remaining_nodes=20});if((string)client.Receive()["type"]!="ready")throw new IOException("handshake");
                        for(int batch=0;batch<3;batch++){
                            var action=client.Receive();if((string)action["type"]!="action")throw new IOException("expected action");var command=action.ToObject<ActionCommand>();bool accepted=world.Deploy(command,out string reason);client.Send(new ObservationPacket{type="action_ack",run_id=run,seq_no=++seq,sim_step=world.Step,timestamp_ms=world.Step*20L,action_id=command.action_id,accepted=accepted});client.Receive();if(!accepted)throw new Exception(reason);
                            Action<ObservationPacket> send=p=>{obs.WriteLine(Wire.Serialize(p));client.Send(p);var response=client.Receive();if(p.batch_end&&(string)response["type"]!="inference")throw new IOException("missing inference");};
                            for(int step=0;step<steps;step++){world.Tick();if(step%10==9)Drain(world,run,command,ref seq,false,send);}Drain(world,run,command,ref seq,true,send);
                        }
                        if((string)client.Receive()["type"]!="result")throw new IOException("missing result");
                        truth.WriteLine(JsonConvert.SerializeObject(Truth(world,run,family,map,seed,sensor,family==5?"ood":"test",policy,loss)));
                    }
                    obs.Flush();truth.Flush();count++;Debug.Log($"COMPARE n={count} family={family} map={map} policy={policy} loss={loss} elapsed={watch.Elapsed.TotalSeconds:F1}s");
                }
                File.WriteAllText(Path.Combine(Output,stem+"-run.json"),JsonConvert.SerializeObject(new{episodes=count,unity=Application.unityVersion,elapsed_seconds=watch.Elapsed.TotalSeconds,steps,repeats,maps},Formatting.Indented));
            }
        }
        static void Smoke(){ResetScene();for(int f=0;f<5;f++)using(var w=new SimulationWorld()){w.Build(f,2026000,2036000,42100);var a=new ActionCommand{entry_id=1,node_count=5,formation_mode="CHAIN_INSERT",policy="fixed"};if(!w.Deploy(a,out string reason))throw new Exception(reason);float start=w.Nodes[0].Body.position.y;for(int s=0;s<450;s++)w.Tick();if(w.Nodes.Count!=5||w.Step!=450||w.Nodes[0].Body.position.y>=start)throw new Exception("gravity/deploy smoke failed");if(!w.Pending.Any(e=>e.kind=="IMPACT")||!w.Pending.Any(e=>e.kind=="PEER_HEARD"))throw new Exception("sensor smoke failed");Debug.Log($"SMOKE PASS family={f} events={w.Pending.Count} depth={w.MaxDepth:F3} couplings={w.Pending.Count(e=>e.kind=="COUPLE_START")}");}}
        static void Build(){Initialize();string output=Arg("-purumoguBuild",Path.GetFullPath(Path.Combine(Application.dataPath,"../../build/PurumoguSearch.exe")));Directory.CreateDirectory(Path.GetDirectoryName(output));var report=BuildPipeline.BuildPlayer(new[]{"Assets/Scenes/Main.unity"},output,BuildTarget.StandaloneWindows64,BuildOptions.None);if(report.summary.result!=UnityEditor.Build.Reporting.BuildResult.Succeeded)throw new Exception("player build failed");Debug.Log("BUILD SUCCEEDED "+output);}
    }
}
