using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Purumogu.Rescue;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Purumogu.Editor
{
    public static class RescueTests
    {
        static string Arg(string name,string fallback){var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]==name)return args[i+1];return fallback;}
        static void Reset()=>EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        public static void AuditSources()
        {
            Reset();var rows=new List<object>();int repeats=int.Parse(Arg("-purumoguRepeats","3")),seedBase=int.Parse(Arg("-purumoguSeedBase","20262000"));
            for(int scene=0;scene<15;scene++)for(int repeat=0;repeat<repeats;repeat++)using(var world=new RescueScene(scene,seedBase+repeat*997))
            {
                for(int i=0;i<world.VictimPositions.Count;i++){var p=world.VictimPositions[i];bool blocked=Physics.CheckSphere(p,.012f,1<<6);if(blocked)throw new Exception("Generated source lies in solid geometry");rows.Add(new{scene=world.Spec.Id,seed=seedBase+repeat*997,person=i,position=RescueRobot.Array(p),point_inside_collider=blocked,placement="independent-seed free-air source point",checked_radius_m=.012f});}
            }
            var output=Arg("-purumoguOutput",Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/source-audit.json")));Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,JsonConvert.SerializeObject(rows,Formatting.Indented));Debug.Log("SOURCE_AUDIT "+output);
        }
        public static void Smoke()
        {
            Reset();if(Mathf.Abs(RescueRobot.PowerRatioDb(.1f,1)+10)>.001f||Mathf.Abs(RescueRobot.PowerRatioDb(1,1))>.001f||Mathf.Abs(RescueRobot.PowerRatioDb(10,1)-10)>.001f)throw new Exception("Acoustic power/dB convention mismatch");CheckClimbingFixture();CheckLostFixture();var records=new List<object>();
            for(int i=0;i<15;i++)using(var sim=new RescueSimulation(i,20261027,6))
            {
                if(sim.Robots.Any(r=>r.Hinges.Any(j=>!j)))throw new Exception("missing physical articulation");
                sim.TryDock();int captured=sim.Attached;if(captured==0)throw new Exception("near aligned ports did not connect");
                sim.ReleaseAll();if(sim.Links.Count!=0||sim.Detached!=captured||sim.Robots.Any(r=>r.PortBusy.Any(b=>b)))throw new Exception("release failed");
                float maxBend=0,maxTravel=0;var initial=sim.Robots.Select(r=>r.Core.position).ToArray();
                for(int step=0;step<750;step++)
                {
                    sim.Tick();if(sim.Step%25==0)sim.Observe("smoke",step/25);for(int n=0;n<sim.Robots.Count;n++){var r=sim.Robots[n];if(!float.IsFinite(r.Core.position.x)||r.Core.position.magnitude>10)throw new Exception($"unstable rigid body {sim.Scene.Spec.Id} {r.Id} {r.Core.position}");maxBend=Mathf.Max(maxBend,r.BendDegrees);maxTravel=Mathf.Max(maxTravel,Vector3.Distance(initial[n],r.Core.position));}
                }
                var packet=sim.Observe("smoke",0);if(packet.scans.Count==0||packet.scans.Any(s=>s.rays.Count<20))throw new Exception("missing range observation");
                if(maxBend<5||maxTravel<.01f)throw new Exception("joint drive or locomotion did not move");
                records.Add(new{scene=sim.Scene.Spec.Id,max_bend_deg=maxBend,max_travel_m=maxTravel,couplings=sim.Attached,releases=sim.Detached,scan_nodes=packet.scans.Count,lost_nodes=sim.Robots.Count(r=>r.Lost)});
                Debug.Log($"RESCUE_SMOKE PASS {sim.Scene.Spec.Id} bend={maxBend:F1} travel={maxTravel:F3} coupling={sim.Attached} release={sim.Detached}");
            }
            string path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue"));Directory.CreateDirectory(path);File.WriteAllText(Path.Combine(path,"physics-smoke.json"),JsonConvert.SerializeObject(records,Formatting.Indented));
        }
        static void CheckClimbingFixture()
        {
            Physics.simulationMode=SimulationMode.Script;Physics.gravity=new Vector3(0,-9.81f,0);
            var floor=GameObject.CreatePrimitive(PrimitiveType.Cube);floor.layer=6;floor.transform.position=new Vector3(0,-.025f,.2f);floor.transform.localScale=new Vector3(1.5f,.05f,1.2f);
            var step=GameObject.CreatePrimitive(PrimitiveType.Cube);step.layer=6;step.transform.position=new Vector3(0,.0125f,.2f);step.transform.localScale=new Vector3(.24f,.025f,.16f);Physics.SyncTransforms();
            bool climbed=false,passed=false;float spin=0;
            using(var robot=new RescueRobot("N00",new Vector3(0,.018f,-.15f),1))
            {
                for(int s=0;s<2500;s++){robot.Motor(s);Physics.Simulate(.02f);robot.AfterPhysics();robot.EstimatedPosition=robot.Core.position;spin=Mathf.Max(spin,robot.SpinDegrees);if(s%100==0)Debug.Log($"CLIMB_TRACE {s} p={robot.Core.position:F4} front={robot.Port(0):F4} rear={robot.Port(1):F4} bend={robot.BendDegrees:F1} rot={robot.Core.rotation.eulerAngles:F1}");if(Mathf.Abs(robot.Core.position.x)<.10f&&robot.Core.position.z>.12f&&robot.Core.position.z<.28f&&robot.Core.position.y>.028f)climbed=true;if(robot.Core.position.z>.4f){passed=true;break;}}
            }
            UnityEngine.Object.DestroyImmediate(floor);UnityEngine.Object.DestroyImmediate(step);
            string path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/climbing-diagnostic.json"));Directory.CreateDirectory(Path.GetDirectoryName(path));File.WriteAllText(path,JsonConvert.SerializeObject(new{step_height_m=.025,climbed,passed,spin_deg=spin,capability_confirmed=climbed&&passed,transform_repositioning=false},Formatting.Indented));if(spin<30)throw new Exception("Axial spin failed");
            Debug.Log($"CLIMBING_DIAGNOSTIC climbed={climbed} passed={passed}; 25mm capability is not a required smoke assertion and is reported separately");
        }
        static void CheckLostFixture()
        {
            using(var robot=new RescueRobot("N00",new Vector3(0,.02f,0),1))
            {
                Physics.SyncTransforms();
                for(int step=0;step<100&&!robot.Lost;step++){Physics.Simulate(.02f);robot.AfterPhysics();}
                if(!robot.Lost||robot.Powered||!robot.Core.isKinematic||robot.Cones.Any(body=>!body.isKinematic))throw new Exception("Dropped robot was not explicitly disabled");
            }
            Debug.Log("LOST_NODE_FIXTURE PASS / gravity drop is logged as loss, no transform relocation");
        }
        public static void Evaluate()
        {
            Reset();int port=int.Parse(Arg("-purumoguPort","47120")),steps=int.Parse(Arg("-purumoguSteps","4500")),repeats=int.Parse(Arg("-purumoguRepeats","2")),seedBase=int.Parse(Arg("-purumoguSeedBase","20261027"));string split=Arg("-purumoguSplit","development");bool guidance=Arg("-purumoguGuidance","false")=="true";
            string path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/"+split));Directory.CreateDirectory(path);string raw=Path.GetFullPath(Path.Combine(Application.dataPath,"../../experiments/raw/rescue"));Directory.CreateDirectory(raw);
            var outcomes=new List<object>();
            File.WriteAllText(Path.Combine(path,"manifest.json"),JsonConvert.SerializeObject(new{status="running",generation="rescue-2.5",expected_maps=15*repeats,steps,seed_base=seedBase,split,guidance},Formatting.Indented));
            using(var training=new StreamWriter(Path.Combine(raw,"features-"+split+".jsonl"),false,new UTF8Encoding(false)))
            using(var grids=new StreamWriter(Path.Combine(raw,"grids-"+split+".jsonl"),false,new UTF8Encoding(false)))
            {
            for(int scene=0;scene<15;scene++)for(int repeat=0;repeat<repeats;repeat++)
            {
                int seed=seedBase+repeat*997;string run=Guid.NewGuid().ToString("N"),mapId="rescue-2.5-"+RescueScene.Catalog[scene].Id+"-"+seed;long seq=0;
                using(var socket=new TcpClient("127.0.0.1",port))using(var sim=new RescueSimulation(scene,seed))
                using(var audit=new StreamWriter(Path.Combine(raw,run+".jsonl"),false,new UTF8Encoding(false)))
                {
                    socket.NoDelay=true;socket.ReceiveTimeout=30000;socket.SendTimeout=30000;var stream=socket.GetStream();var reader=new StreamReader(stream,Encoding.UTF8);var writer=new StreamWriter(stream,new UTF8Encoding(false)){AutoFlush=true};var evaluation=new RescueEvaluation();
                    JObject finalMap=null;
                    for(int step=0;step<steps;step++)
                    {
                        sim.Tick();if(sim.Step%25!=0)continue;
                        var packet=sim.Observe(run,seq++);packet.request_map=sim.Step+25>steps;string json=JsonConvert.SerializeObject(packet);audit.WriteLine(json);writer.WriteLine(json);string line=reader.ReadLine();if(line==null||line.Length>8388608)throw new IOException("bad map response");var reply=JObject.Parse(line);if((string)reply["type"]=="error")throw new IOException(reply.ToString());if((long)reply["seq_no"]!=packet.seq_no)throw new IOException("sequence mismatch");if(sim.Step==25||sim.Step+25>steps)audit.WriteLine(line);if(guidance)sim.ApplyGuidance(reply);if(packet.request_map)evaluation.Evaluate(sim,reply);
                        finalMap=reply;
                        foreach(var scan in packet.scans){var robot=sim.Robots.Find(r=>r.Id==scan.node_id);training.WriteLine(JsonConvert.SerializeObject(new{kind="acoustic",map_id=mapId,split,features=new[]{scan.acoustic_snr_db/20,scan.voice_band_ratio,scan.modulation,scan.temperature_delta_c/2},label=robot.EvaluationSourceIsHuman?1:0}));}
                    }
                    if(finalMap!=null)foreach(var c in (JArray)finalMap["cells"])
                    {
                        var k=(JArray)c["key"];var p=new Vector3(((int)k[0]+.5f)*RescueScene.Cell,((int)k[1]+.5f)*RescueScene.Cell,((int)k[2]+.5f)*RescueScene.Cell);bool truth=Physics.CheckBox(p,Vector3.one*(RescueScene.Cell*.42f),Quaternion.identity,1<<6);float prob=(float)c["occupancy"];training.WriteLine(JsonConvert.SerializeObject(new{kind="occupancy",map_id=mapId,split,features=new[]{Mathf.Log(prob/(1-prob)),Mathf.Log(1+(int)c["observations"])},label=truth?1:0}));
                    }
                    var row=JObject.FromObject(evaluation.Snapshot(sim));row["seed"]=seed;row["repeat"]=repeat;row["map_id"]=mapId;row["run_id"]=run;row["split"]=split;outcomes.Add(row);File.WriteAllText(Path.Combine(path,"outcomes.json"),JsonConvert.SerializeObject(outcomes,Formatting.Indented));training.Flush();
                    grids.WriteLine(JsonConvert.SerializeObject(new{scene=sim.Scene.Spec.Id,map_id=mapId,run_id=run,split,occupied_keys=evaluation.TrainingTruthGrid(),victim_positions=sim.Scene.VictimPositions.Select(RescueRobot.Array).ToArray()}));grids.Flush();
                    Debug.Log($"RESCUE_EVAL {sim.Scene.Spec.Id} repeat={repeat} IoU={evaluation.IoU:F3} coverage={evaluation.Coverage:F3} targets={evaluation.Matched}/{sim.Scene.Spec.Victims} false={evaluation.FalseAlarms}");
                }
            }
            }
            File.WriteAllText(Path.Combine(path,"manifest.json"),JsonConvert.SerializeObject(new{status="complete",generation="rescue-2.5",expected_maps=15*repeats,unity=Application.unityVersion,scenes=15,repeats,steps,seconds=steps*.02,nodes=12,seed_base=seedBase,split,guidance,robot_length_m=.07,robot_mass_kg=.024,cell_m=RescueScene.Cell,inference="Bayesian occupancy / independent-node acoustic bearing fusion",source_placement="independent-position-v1",acoustic_power_db="10log10; all-source power mixture; no human label in observations",truth_to_server=false,targets=new{global_occupancy_iou=.95,human_precision=.95,human_recall=.95}},Formatting.Indented));
        }
    }
}
