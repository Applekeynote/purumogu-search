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
    /// Offline labels and physical rollouts have separate files. Only RescuePacket reaches Rust.
    public static class RescueGeneralization
    {
        static string Arg(string key,string fallback){var a=Environment.GetCommandLineArgs();for(int i=0;i<a.Length-1;i++)if(a[i]==key)return a[i+1];return fallback;}
        public static void Collect()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            var plan=JObject.Parse(File.ReadAllText(Path.GetFullPath(Arg("-purumoguPlan","../experiments/generalization-v35/plan.json"))));
            string output=Path.GetFullPath((string)plan["output"]);Directory.CreateDirectory(output);
            string collectionFile=(string)plan["collection_file"]??"collection.json";
            if(Path.GetFileName(collectionFile)!=collectionFile||!collectionFile.EndsWith(".json",StringComparison.Ordinal))throw new Exception("Invalid collection manifest filename");
            int port=int.Parse(Arg("-purumoguPort","47120")),steps=(int)plan["steps"];
            int mapInterval=(int?)plan["inference_map_interval"]??1500;
            if(mapInterval<25||mapInterval>1500||mapInterval%25!=0)throw new Exception("Invalid inference map interval");
            string split=(string)plan["split"];if(split.StartsWith("test",StringComparison.Ordinal)&&Arg("-purumoguFinalTest","false")!="true")throw new Exception("Frozen TEST requires explicit final-test switch");
            var outcomes=new List<object>();
            foreach(var item in (JArray)plan["cases"])
            {
                int scene=(int)item["scene"],seed=(int)item["seed"],nodes=(int)item["nodes"],deployment=(int)item["deployment_seed"],source=(int)item["source_seed"];
                string id=(string)item["id"],run=Guid.NewGuid().ToString("N"),directory=Path.Combine(output,id);
                // Resume completed immutable cases, never replace an earlier audit.
                if(File.Exists(Path.Combine(directory,"complete.json"))){Debug.Log("GENERALIZATION_SKIP "+id);continue;}
                if(Directory.Exists(directory))throw new Exception("Incomplete case exists; retain it and use a new output directory: "+directory);
                Directory.CreateDirectory(directory);
                string modeName=(string)item["deployment_mode"]??(string)plan["deployment_mode"]??"WholeSceneDrop";
                if(!Enum.TryParse(modeName,out RescueDeploymentMode mode)||mode==RescueDeploymentMode.EntranceGrid)throw new Exception("Invalid scattering mode");
                using(var sim=new RescueSimulation(scene,seed,nodes,mode,deployment,source))
                using(var socket=new TcpClient("127.0.0.1",port))
                using(var audit=new StreamWriter(Path.Combine(directory,"packets.jsonl"),false,new UTF8Encoding(false)))
                using(var labels=new StreamWriter(Path.Combine(directory,"labels.jsonl"),false,new UTF8Encoding(false)))
                {
                    socket.NoDelay=true;socket.ReceiveTimeout=120000;socket.SendTimeout=120000;
                    var stream=socket.GetStream();var reader=new StreamReader(stream,Encoding.UTF8);var writer=new StreamWriter(stream,new UTF8Encoding(false)){AutoFlush=true};
                    JObject reply=null;long sequence=0;var evaluation=new RescueEvaluation();
                    for(int step=0;step<steps;step++)
                    {
                        sim.Tick();if(sim.Step%25!=0)continue;
                        bool snapshot=sim.Step%1500==0||sim.Step==steps;
                        var packet=sim.Observe(run,sequence++);packet.request_map=snapshot||sim.Step%mapInterval==0;packet.request_diagnostics=true;
                        var text=JsonConvert.SerializeObject(packet);audit.WriteLine(text);writer.WriteLine(text);
                        string line=reader.ReadLine();if(line==null||line.Length>8388608)throw new IOException("Invalid inference response");
                        reply=JObject.Parse(line);if((string)reply["type"]=="error")throw new IOException(line);
                        sim.ApplyGuidance(reply);
                        labels.WriteLine(JsonConvert.SerializeObject(new{step=sim.Step,robots=sim.Robots.Select(r=>new{id=r.Id,position=RescueRobot.Array(r.Core.position),lost=r.Lost,released=r.Released,human=r.EvaluationSourceIsHuman}).ToArray()}));
                        if(snapshot)
                        {
                            evaluation.Evaluate(sim,reply);
                            File.WriteAllText(Path.Combine(directory,"map-"+sim.Step+".json"),line);
                            File.WriteAllText(Path.Combine(directory,"truth-"+sim.Step+".json"),JsonConvert.SerializeObject(new{occupied_keys=evaluation.TrainingTruthGrid(),victim_positions=sim.Scene.VictimPositions.Select(RescueRobot.Array).ToArray(),metrics=evaluation.Snapshot(sim)}));
                        }
                    }
                    var result=JObject.FromObject(evaluation.Snapshot(sim));result["id"]=id;result["split"]=split;result["case"]=item.DeepClone();result["unity"]=Application.unityVersion;result["truth_to_server"]=false;result["guidance"]=true;
                    result["acquisition"]=new JObject{["model_sha256"]=plan["model_sha256"]?.DeepClone(),["server_sha256"]=plan["server_sha256"]?.DeepClone(),["geometry_revision"]=plan["geometry_revision"]?.DeepClone(),["sensor_source_sha256"]=plan["sensor_source_sha256"]?.DeepClone(),["inference_map_interval"]=mapInterval};
                    audit.Flush();labels.Flush();File.WriteAllText(Path.Combine(directory,"complete.json"),result.ToString(Formatting.Indented));outcomes.Add(result);
                    Debug.Log("GENERALIZATION_COMPLETE "+id+" IoU="+evaluation.GlobalIoU+" targets="+evaluation.Matched+"/"+sim.Scene.VictimPositions.Count);
                }
            }
            File.WriteAllText(Path.Combine(output,collectionFile),JsonConvert.SerializeObject(new{status="complete",split,steps,plan,truth_to_server=false,outcomes},Formatting.Indented));
        }
    }
}
