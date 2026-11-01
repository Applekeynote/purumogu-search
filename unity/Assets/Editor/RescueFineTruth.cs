using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Purumogu.Rescue;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Purumogu.Editor
{
    /// Dense TRAIN supervision only. The deployed inference never receives these labels.
    public static class RescueFineTruth
    {
        static string Arg(string key){var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]==key)return args[i+1];throw new ArgumentException(key);}
        static HashSet<Vector3Int> Grid(int subdivision)
        {
            var result=new HashSet<Vector3Int>();float cell=.05f/subdivision;
            for(int x=-16*subdivision;x<16*subdivision;x++)for(int y=-3*subdivision;y<18*subdivision;y++)for(int z=-6*subdivision;z<64*subdivision;z++)
                if(Physics.CheckBox(new Vector3((x+.5f)*cell,(y+.5f)*cell,(z+.5f)*cell),Vector3.one*(cell*.42f),Quaternion.identity,1<<6))result.Add(new Vector3Int(x,y,z));
            return result;
        }
        public static void Export()
        {
            try
            {
                var plan=JObject.Parse(File.ReadAllText(Arg("-purumoguPlan")));
                if((string)plan["split"]!="train"||(bool)plan["test_opened"])throw new Exception("TRAIN labels only");
                string output=(string)plan["output"];Directory.CreateDirectory(output);
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
                int count=0;
                foreach(var item in (JArray)plan["cases"])
                {
                    string id=(string)item["id"],directory=Path.Combine(output,id);int scene=(int)item["scene"];
                    if(new[]{17,19,22,23,27,28,29,34,35}.Contains(scene))throw new Exception("Held family entered TRAIN labels");
                    if(Directory.Exists(directory))throw new Exception("Immutable fine labels already exist: "+id);
                    using(var world=new RescueScene(scene,(int)item["seed"],(int)item["source_seed"]))
                    {
                        if(world.LoosePieces.Count>0)throw new Exception("Moving geometry needs its exact checkpoint: "+id);
                        world.ConfigureExteriorDeployment((int)item["seed"]);Physics.SyncTransforms();
                        var original=JObject.Parse(File.ReadAllText((string)item["source_truth"]));
                        var expected=new HashSet<Vector3Int>(((JArray)original["occupied_keys"]).Select(k=>new Vector3Int((int)k[0],(int)k[1],(int)k[2])));
                        if(!Grid(1).SetEquals(expected))throw new Exception("Recorded 5cm TRAIN geometry differs: "+id);
                        var fine=Grid(2);Directory.CreateDirectory(directory);
                        File.WriteAllText(Path.Combine(directory,"fine-truth.json"),JsonConvert.SerializeObject(new{cell_m=.025,minimum_key=new[]{-32,-6,-12},shape=new[]{64,42,140},occupied_keys=fine.OrderBy(k=>k.x).ThenBy(k=>k.y).ThenBy(k=>k.z).Select(k=>new[]{k.x,k.y,k.z}).ToArray()}));
                        File.WriteAllText(Path.Combine(directory,"complete.json"),JsonConvert.SerializeObject(new{id,split="train",coarse_geometry_exact=true,fine_cells=fine.Count,truth_to_inference=false,source_truth=(string)item["source_truth"]},Formatting.Indented));
                    }
                    if(++count%12==0)Debug.Log("FINE_TRUTH "+count+"/"+((JArray)plan["cases"]).Count);
                }
                File.WriteAllText(Path.Combine(output,"collection.json"),JsonConvert.SerializeObject(new{status="complete",split="train",test_opened=false,plan,cases=count},Formatting.Indented));
                Debug.Log("FINE_TRUTH_COMPLETE "+count);EditorApplication.Exit(0);
            }
            catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}
        }
    }
}
