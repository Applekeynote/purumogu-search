using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Purumogu.Rescue;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Purumogu.Editor
{
    /// Checks the physical passage of connecting ramps; no inference accuracy is scored here.
    public static class RescueConnectivity
    {
        public static void Audit()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            var results=new List<object>();int blocked=0;
            // Final unseen-family TEST worlds are deliberately not instantiated.
            foreach(int index in new[]{15,16,18,19,20,21,22,24,25,26,29})
            {
                int seed=41000000+index*1009;
                using(var scene=new RescueScene(index,seed,seed+900000))
                {
                    Physics.SyncTransforms();
                    foreach(var ramp in scene.Root.GetComponentsInChildren<BoxCollider>().Where(c=>c.name.StartsWith("ConnectingRamp_")))
                    {
                        var obstructions=new HashSet<string>();int samples=0;
                        for(int step=2;step<=38;step++)
                        {
                            float t=step/40f;
                            var point=ramp.transform.TransformPoint(new Vector3(0,.5f,t-.5f))+Vector3.up*.012f;
                            foreach(var collider in Physics.OverlapSphere(point,.008f,1<<6))
                            {
                                if(!collider.name.StartsWith("ShapedFloor_"))continue;
                                // A small landing step is traversable by the existing motor;
                                // a slab above the robot's head is a blocked flight.
                                if(point.y<collider.bounds.max.y-.033f){samples++;obstructions.Add(collider.name);}
                            }
                        }
                        if(samples>0)blocked++;
                        results.Add(new{scene=scene.Spec.Id,ramp=ramp.name,blocked_samples=samples,obstructions=obstructions.OrderBy(s=>s).ToArray(),debris_count=scene.Root.GetComponentsInChildren<Transform>().Count(t=>t.name.StartsWith("LevelDebris_"))});
                    }
                }
            }
            string output=Path.GetFullPath("../reports/rescue/ramp-connectivity-audit.json");
            var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]=="-purumoguOutput")output=Path.GetFullPath(args[i+1]);
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            File.WriteAllText(output,JsonConvert.SerializeObject(new{purpose="Physical ramp clearance audit; TRAIN/VALID scenes only",test_opened=false,blocked_flights=blocked,results},Formatting.Indented));
            Debug.Log("RAMP_CONNECTIVITY_AUDIT blocked="+blocked+" output="+output);
        }
    }
}
