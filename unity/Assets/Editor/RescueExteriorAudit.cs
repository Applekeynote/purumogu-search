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
    public static class RescueExteriorAudit
    {
        public static void Audit()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
            var rows=new List<object>();int blocked=0,sideLost=0,totalSide=0;
            foreach(int index in new[]{0,3,8,13,15,16,18,19,20,21,22,24,25,26,29})foreach(int count in new[]{24,48})
            {
                int seed=61000000+index*1009,drop=seed+count*131+700000;
                using(var simulation=new RescueSimulation(index,seed,count,RescueDeploymentMode.RoofAndFacadeDrop,drop,seed+900000))
                {
                    var scene=simulation.Scene;var sites=scene.ExteriorOpenings;
                    var releases=RescueDeployment.Plan(count,drop,RescueDeploymentMode.RoofAndFacadeDrop,scene.ComplexGeometry?scene.DeploymentRegions:null,sites);
                    var blockedSites=new List<object>();
                    foreach(var site in sites)
                    {
                        var start=site.FloorPoint+site.Outward*.10f+Vector3.up*.15f;
                        var end=site.FloorPoint-site.Outward*.10f+Vector3.up*.15f;
                        var obstructions=Physics.OverlapCapsule(start,end,.010f,1<<6).Where(c=>c.name.StartsWith("ExteriorWallRemnant")||c.name.StartsWith("FootprintWall")||c.name.StartsWith("BrokenSide")).Select(c=>c.name).ToArray();
                        if(obstructions.Length>0){blocked++;blockedSites.Add(new{position=RescueRobot.Array(site.FloorPoint),site.Level,site.HalfWall,obstructions});}
                    }
                    simulation.Power(false);for(int step=0;step<850;step++)simulation.Tick();
                    var side=Enumerable.Range(0,count).Where(i=>releases[i].Velocity!=Vector3.zero).ToArray();
                    int lost=side.Count(i=>simulation.Robots[i].Lost);sideLost+=lost;totalSide+=side.Length;
                    var settled=new Dictionary<int,int>();
                    for(int level=0;level<scene.FloorCount;level++)settled[level]=side.Count(i=>!simulation.Robots[i].Lost&&Mathf.Abs(simulation.Robots[i].Core.position.y-level*.28f)<.12f);
                    rows.Add(new{scene=scene.Spec.Id,count,seed,openings=sites.Select(s=>new{level=s.Level,position=RescueRobot.Array(s.FloorPoint),outward=RescueRobot.Array(s.Outward)}).ToArray(),roof_nodes=count-side.Length,side_nodes=side.Length,side_lost=lost,settled_per_level=settled,
                        gravity=Physics.gravity.y,launch_speed_m_s=RescueDeployment.SideLaunchSpeed,sensors_modified=false,sources_used_for_release=false,blocked_sites=blockedSites,
                        debris_count=scene.Root.GetComponentsInChildren<Transform>().Count(t=>t.name.StartsWith("LevelDebris_"))});
                }
            }
            string output=Path.GetFullPath("../reports/rescue/exterior-deployment-audit.json");var args=Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]=="-purumoguOutput")output=Path.GetFullPath(args[i+1]);
            Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,JsonConvert.SerializeObject(new{test_opened=false,purpose="Physical external release audit; TRAIN scenes, motors off, natural collision and gravity",blocked_openings=blocked,side_lost=sideLost,total_side_nodes=totalSide,rows},Formatting.Indented));
            if(blocked>0)throw new Exception("An exterior release opening is still blocked");
            if(sideLost>totalSide*.1f)throw new Exception("External launch loses more than 10% of side-deployed nodes before exploration");
        }
    }
}
