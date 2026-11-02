using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Purumogu.Rescue;
using UnityEngine;

namespace Purumogu.Editor
{
    public static class RescueRenderTests
    {
        static void Finish(Func<bool> present,Func<bool> complete)
        {
            var limit=DateTime.UtcNow.AddSeconds(10);
            while(!complete()){present();if(DateTime.UtcNow>limit)throw new Exception("Background display timed out");Thread.Sleep(1);}
        }
        public static void Smoke()
        {
            RescueTests.DashboardSmoke();
            var frames=new List<JObject>();
            for(int frame=0;frame<4;frame++)
            {
                var cells=new JArray();
                for(int x=-4;x<=4;x++)for(int y=-2;y<=4;y++)for(int z=-4;z<=4;z++)
                {
                    bool solid=Math.Abs(x)<=2&&y>=0&&y<=2&&Math.Abs(z)<=2;
                    float sensor=solid?.95f:.05f;
                    if(frame==3&&x==2)sensor=.05f;
                    cells.Add(JObject.FromObject(new{key=new[]{x,y,z},occupancy=solid?.95f:.05f,sensor_occupancy=sensor,observations=frame==0?1:frame==3?4:2}));
                }
                frames.Add(JObject.FromObject(new{cell_m=.05,completion_family="volume-unet-v1",cells,cues=new[]{new{position=new[]{0,.05,0},radius_m=.18}},hazards=new[]{new{position=new[]{.1,.05,.1},radius_m=.12,kind="gas"},new{position=new[]{-.1,.05,-.1},radius_m=.12,kind="water"}}}));
            }
            using(var reference=new RescueEchoWireMap())using(var threaded=new RescueEchoWireMap())
            {
                foreach(var frame in frames){reference.Update(frame);threaded.ScheduleUpdate(frame);}
                int applied=0;Finish(()=>{if(threaded.PresentCompleted())applied++;return false;},()=>applied==frames.Count);
                if(reference.Quads!=threaded.Quads||reference.Segments!=threaded.Segments||reference.InferredCells!=threaded.InferredCells||reference.PersonSegments!=threaded.PersonSegments||reference.GasSegments!=threaded.GasSegments||reference.WaterSegments!=threaded.WaterSegments)throw new Exception("Background map changed geometry or hazard colors");
                var a=reference.Root.GetComponentsInChildren<MeshFilter>();var b=threaded.Root.GetComponentsInChildren<MeshFilter>();
                for(int i=0;i<a.Length;i++)
                {
                    var x=a[i].sharedMesh;var y=b[i].sharedMesh;
                    if(!x.vertices.SequenceEqual(y.vertices)||!x.triangles.SequenceEqual(y.triangles)||!x.colors.SequenceEqual(y.colors)||!x.uv.SequenceEqual(y.uv)||!x.uv2.SequenceEqual(y.uv2)||!x.normals.SequenceEqual(y.normals))throw new Exception("Background mesh buffers differ");
                }
            }
            var discarded=new RescueEchoWireMap();discarded.ScheduleUpdate(frames[0]);discarded.Dispose();if(discarded.PresentCompleted())throw new Exception("Disposed map published stale data");
            using(var sim=new RescueSimulation(0,20261027,2))
            {
                var motion=new RescueVisualMotion(sim.Robots);
                var before=sim.Robots[0].Core.position;
                foreach(var robot in sim.Robots)foreach(var body in new[]{robot.Core,robot.Cones[0],robot.Cones[1]})body.position+=new Vector3(.01f,.005f,0);
                motion.Capture();Physics.SyncTransforms();
                var bodies=sim.Robots.SelectMany(r=>new[]{r.Core,r.Cones[0],r.Cones[1]}).ToArray();
                var positions=bodies.Select(b=>b.position).ToArray();var rotations=bodies.Select(b=>b.rotation).ToArray();
                var ports=sim.Robots.SelectMany(r=>Enumerable.Range(0,RescueRobot.PortCount).Select(r.Port)).ToArray();
                var colliders=sim.Robots.SelectMany(r=>r.Root.GetComponentsInChildren<Collider>()).ToArray();var bounds=colliders.Select(c=>c.bounds).ToArray();
                motion.PresentFraction(.5f);Physics.SyncTransforms();
                if(Vector3.Distance(sim.Robots[0].Visuals[0].position,(before+positions[0])*.5f)>1e-6f)throw new Exception("Visual pose was not interpolated");
                foreach(float fraction in new[]{0f,.25f,.75f,1f})motion.PresentFraction(fraction);
                Physics.SyncTransforms();
                if(!positions.SequenceEqual(bodies.Select(b=>b.position))||!rotations.SequenceEqual(bodies.Select(b=>b.rotation))||!ports.SequenceEqual(sim.Robots.SelectMany(r=>Enumerable.Range(0,RescueRobot.PortCount).Select(r.Port)))||!bounds.SequenceEqual(colliders.Select(c=>c.bounds)))throw new Exception("Visual interpolation moved physics, sensors or dock ports");
                var reply=(JObject)frames[2].DeepClone();reply["sim_step"]=sim.Step;
                reply["nodes"]=new JArray(sim.Robots.Select(r=>JObject.FromObject(new{node_id=r.Id,estimated_position=RescueRobot.Array(r.Core.position+new Vector3(.01f,0,0))})));
                reply["cues"]=new JArray(sim.Scene.VictimPositions.Select(p=>JObject.FromObject(new{position=RescueRobot.Array(p)})));
                var reference=new RescueEvaluation();var threaded=new RescueEvaluation();reference.Evaluate(sim,reply);threaded.Schedule(sim,reply);
                var expected=JsonConvert.SerializeObject(reference.Snapshot(sim));
                // Later physics positions must not leak into an already captured evaluation.
                foreach(var robot in sim.Robots)robot.Core.position+=Vector3.right*.1f;
                Finish(threaded.PresentCompleted,()=>!threaded.Pending);
                if(expected!=JsonConvert.SerializeObject(threaded.Snapshot(sim)))throw new Exception("Asynchronous evaluation changed metric values or pose snapshot");
                threaded.Schedule(sim,reply);threaded.ResetDisplay();Thread.Sleep(25);threaded.PresentCompleted();
                if(threaded.Pending||threaded.Observed!=0||threaded.GlobalIoU!=0)throw new Exception("Reset accepted stale evaluation");
            }
            Debug.Log("RESCUE_RENDER_SMOKE PASS: ordered mesh buffers, colors, metric parity, pose snapshot, reset/dispose, visual interpolation preserves physics and ports");
        }
    }
}
