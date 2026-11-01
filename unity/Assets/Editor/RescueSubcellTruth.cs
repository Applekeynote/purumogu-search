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
    /// TRAIN labels only. No labels or true poses enter the inference process.
    public static class RescueSubcellTruth
    {
        static string Arg(string key){var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]==key)return args[i+1];throw new ArgumentException(key);}
        static HashSet<Vector3Int> Grid(Vector3 translation,int subdivision=1)
        {
            var result=new HashSet<Vector3Int>();float cell=.05f/subdivision;
            for(int x=-16*subdivision;x<16*subdivision;x++)for(int y=-3*subdivision;y<18*subdivision;y++)for(int z=-6*subdivision;z<64*subdivision;z++)
                if(Physics.CheckBox(new Vector3((x+.5f)*cell,(y+.5f)*cell,(z+.5f)*cell)-translation,Vector3.one*(cell*.42f),Quaternion.identity,1<<6))result.Add(new Vector3Int(x,y,z));
            return result;
        }
        public static void Export()
        {
            try
            {
                string planFile=Arg("-purumoguPlan"),planSha;
                byte[] planBytes=File.ReadAllBytes(planFile);
                using(var hash=System.Security.Cryptography.SHA256.Create())planSha=BitConverter.ToString(hash.ComputeHash(planBytes)).Replace("-","").ToLowerInvariant();
                var plan=JObject.Parse(System.Text.Encoding.UTF8.GetString(planBytes).TrimStart('\uFEFF'));
                if((string)plan["split"]!="train"||(bool?)plan["test_opened"]==true)throw new Exception("TRAIN-only coordinate augmentation");
                bool fineSupervision=(bool?)plan["fine_supervision"]==true;
                string output=(string)plan["output"];Directory.CreateDirectory(output);
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
                var outcomes=new List<object>();int count=0;
                foreach(var item in (JArray)plan["cases"])
                {
                    string id=(string)item["id"],directory=Path.Combine(output,id);int scene=(int)item["scene"];
                    if(new[]{17,19,22,23,27,28,29,34,35}.Contains(scene))throw new Exception("Held family entered TRAIN augmentation");
                    if(File.Exists(Path.Combine(directory,"complete.json")))throw new Exception("Immutable augmentation already exists: "+id);
                    using(var world=new RescueScene(scene,(int)item["seed"],(int)item["source_seed"]))
                    {
                        if(world.LoosePieces.Count>0)throw new Exception("Moving geometry needs its exact checkpoint; exclude it: "+id);
                        world.ConfigureExteriorDeployment((int)item["seed"]);Physics.SyncTransforms();
                        var original=JObject.Parse(File.ReadAllText((string)item["source_truth"]));
                        var expected=new HashSet<Vector3Int>(((JArray)original["occupied_keys"]).Select(k=>new Vector3Int((int)k[0],(int)k[1],(int)k[2])));
                        var zero=Grid(Vector3.zero);
                        if(!zero.SetEquals(expected))throw new Exception("Zero-translation geometry differs from recorded TRAIN: "+id+" expected="+expected.Count+" actual="+zero.Count);
                        Directory.CreateDirectory(directory);var views=new List<object>();
                        foreach(var view in (JArray)item["views"])
                        {
                            int index=(int)view["index"];var delta=RescueEvaluation.Vector(view["translation_m"]);
                            if(Mathf.Max(Mathf.Abs(delta.x),Mathf.Abs(delta.y),Mathf.Abs(delta.z))>=.025f)throw new Exception("Subcell translation bound");
                            var truth=Grid(delta);string file="truth-subcell-"+index+".json";
                            File.WriteAllText(Path.Combine(directory,file),JsonConvert.SerializeObject(new{occupied_keys=truth.OrderBy(k=>k.x).ThenBy(k=>k.y).ThenBy(k=>k.z).Select(k=>new[]{k.x,k.y,k.z}).ToArray(),victim_positions=world.VictimPositions.Select(v=>RescueRobot.Array(v+delta)).ToArray(),translation_m=RescueRobot.Array(delta)}));
                            string fineFile=null;
                            if(fineSupervision)
                            {
                                fineFile="fine-truth-subcell-"+index+".json";var fine=Grid(delta,2);
                                File.WriteAllText(Path.Combine(directory,fineFile),JsonConvert.SerializeObject(new{cell_m=.025,minimum_key=new[]{-32,-6,-12},shape=new[]{64,42,140},occupied_keys=fine.OrderBy(k=>k.x).ThenBy(k=>k.y).ThenBy(k=>k.z).Select(k=>new[]{k.x,k.y,k.z}).ToArray()}));
                            }
                            views.Add(new{index,translation_m=RescueRobot.Array(delta),truth=file,fine_truth=fineFile});
                        }
                        var record=new{id,split="train",source_truth=(string)item["source_truth"],zero_translation_exact=true,occupied_cells=zero.Count,views};
                        File.WriteAllText(Path.Combine(directory,"complete.json"),JsonConvert.SerializeObject(record,Formatting.Indented));outcomes.Add(record);
                    }
                    if(++count%12==0)Debug.Log("SUBCELL_TRUTH "+count+"/"+((JArray)plan["cases"]).Count);
                }
                File.WriteAllText(Path.Combine(output,"collection.json"),JsonConvert.SerializeObject(new{status="complete",split="train",test_opened=false,plan_sha256=planSha,plan,outcomes},Formatting.Indented));
                Debug.Log("SUBCELL_TRUTH_COMPLETE "+count);EditorApplication.Exit(0);
            }
            catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}
        }
    }
}
