using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Purumogu.Rescue;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Purumogu.Editor
{
    public static class RescueCollapseAudit
    {
        static string Arg(string key){var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]==key)return args[i+1];throw new ArgumentException(key);}
        static HashSet<Vector3Int> Grid()
        {
            var grid=new HashSet<Vector3Int>();
            for(int x=-16;x<16;x++)for(int y=-3;y<18;y++)for(int z=-6;z<64;z++)
                if(Physics.CheckBox(new Vector3((x+.5f)*.05f,(y+.5f)*.05f,(z+.5f)*.05f),Vector3.one*.021f,Quaternion.identity,1<<6))grid.Add(new Vector3Int(x,y,z));
            return grid;
        }
        static string Digest(IEnumerable<Vector3Int> grid)
        {
            var text=string.Join(";",grid.OrderBy(p=>p.x).ThenBy(p=>p.y).ThenBy(p=>p.z).Select(p=>$"{p.x},{p.y},{p.z}"));
            using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(text))).Replace("-","").ToLowerInvariant();
        }
        public static void Run()
        {
            try
            {
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
                string output=Path.GetFullPath(Arg("-purumoguOutput")),baseline=Path.GetFullPath(Arg("-purumoguBaseline"));
                var geometry=new List<object>();var releases=new List<object>();var regressions=new List<object>();
                foreach(int scene in Enumerable.Range(30,6))foreach(int variant in Enumerable.Range(0,4))
                {
                    int seed=121000000+scene*1009+variant*101;string first=null;Vector3[] originalPeople=null;
                    foreach(int sourceOffset in new[]{900000,1900000})using(var world=new RescueScene(scene,seed,seed+sourceOffset))
                    {
                        world.ConfigureExteriorDeployment(seed);Physics.SyncTransforms();
                        string hash=Digest(Grid());
                        if(first!=null)
                        {
                            if(first!=hash)throw new Exception("Changing source locations changed collision geometry");
                            if(originalPeople.Length>0&&originalPeople.Zip(world.VictimPositions,(a,b)=>Vector3.Distance(a,b)).All(d=>d<.001f))throw new Exception("Sources did not move independently");
                            continue;
                        }
                        first=hash;originalPeople=world.VictimPositions.ToArray();
                        if(world.CollapseLayers.Count!=world.FloorCount-1)throw new Exception("Missing collapse storey");
                        foreach(var layer in world.CollapseLayers)
                        {
                            if(layer.MissingArea<=0||layer.SurvivingArea<=0||layer.FallenSlabs.Count==0)throw new Exception("No genuine partial collapse");
                            if(layer.FallenSlabs.Any(g=>g.transform.position.y>=layer.OriginalHeight-.025f))throw new Exception("Slab did not drop");
                        }
                        foreach(var p in world.VictimPositions)if(Physics.CheckSphere(p,.012f,1<<6))throw new Exception("Person inside solid rubble");
                        geometry.Add(new{scene=world.Spec.Id,seed,storeys=world.FloorCount,geometry_sha256=hash,source_independence=true,openings=world.ExteriorOpenings.Count,
                            layers=world.CollapseLayers.Select(l=>new{l.Level,l.OriginalHeight,l.MissingArea,l.SurvivingArea,slabs=l.FallenSlabs.Count,heights=l.FallenSlabs.Select(g=>g.transform.position.y).ToArray()}).ToArray()});
                    }
                }
                // Old development worlds must remain bit-exact after appending
                // the new catalogue; no old labels or acquisition are rewritten.
                foreach(string split in new[]{"valid-known","valid-unknown"})
                {
                    var plan=JObject.Parse(File.ReadAllText(Path.Combine(baseline,split,"collection.json")));
                    foreach(var item in ((JArray)plan["plan"]["cases"]).Where(c=>new[]{0,15,25,29}.Contains((int)c["scene"])&&(int)c["nodes"]==24))
                    using(var world=new RescueScene((int)item["scene"],(int)item["seed"],(int)item["source_seed"]))
                    {
                        world.ConfigureExteriorDeployment((int)item["seed"]);Physics.SyncTransforms();
                        string id=(string)item["id"];var truth=JObject.Parse(File.ReadAllText(Path.Combine(baseline,split,id,"truth-18000.json")));
                        var expected=new HashSet<Vector3Int>(((JArray)truth["occupied_keys"]).Select(k=>new Vector3Int((int)k[0],(int)k[1],(int)k[2])));
                        if(!Grid().SetEquals(expected))throw new Exception("Existing benchmark geometry changed: "+id);
                        regressions.Add(new{id,coarse_geometry_exact=true});
                    }
                }
                foreach(int scene in Enumerable.Range(30,6))foreach(int count in new[]{24,36,48})
                {
                    int seed=122000000+scene*1009,drop=seed+count*131+700000;
                    using(var sim=new RescueSimulation(scene,seed,count,RescueDeploymentMode.RoofAndFacadeDrop,drop,seed+900000))
                    {
                        var plan=RescueDeployment.Plan(count,drop,RescueDeploymentMode.RoofAndFacadeDrop,sim.Scene.DeploymentRegions,sim.Scene.ExteriorOpenings);
                        sim.Power(false);for(int step=0;step<850;step++)sim.Tick();
                        var side=Enumerable.Range(0,count).Where(i=>plan[i].Velocity!=Vector3.zero).ToArray();
                        var levels=Enumerable.Range(0,sim.Scene.FloorCount).ToDictionary(l=>l,l=>side.Count(i=>!sim.Robots[i].Lost&&Mathf.Abs(sim.Robots[i].Core.position.y-sim.Scene.FloorElevation(l))<.12f));
                        if(levels.Values.Count(v=>v>0)<2)throw new Exception("Release did not reach multiple storeys");
                        releases.Add(new{scene=sim.Scene.Spec.Id,count,floor_heights=Enumerable.Range(0,sim.Scene.FloorCount).Select(sim.Scene.FloorElevation).ToArray(),side_nodes=side.Length,side_lost=side.Count(i=>sim.Robots[i].Lost),settled_per_level=levels,steps=850,seconds=17,sensors_modified=false,source_guided_release=false});
                    }
                    Debug.Log("COLLAPSE_RELEASE_AUDIT "+scene+" N"+count);
                }
                Directory.CreateDirectory(Path.GetDirectoryName(output));
                File.WriteAllText(output,JsonConvert.SerializeObject(new{status="passed",purpose="Post-collapse geometry and passive release; not inference accuracy",test_opened=false,geometry,regressions,releases},Formatting.Indented));
                Debug.Log("COLLAPSE_AUDIT_PASS");EditorApplication.Exit(0);
            }
            catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}
        }
    }
}
