using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed class RescueSimulation : IDisposable
    {
        public readonly RescueScene Scene;
        public readonly List<RescueRobot> Robots=new List<RescueRobot>();
        public readonly List<DockLink> Links=new List<DockLink>();
        public int Step, Attached, Detached;
        public bool Docking=true;
        public readonly RescueDeploymentMode DeploymentMode;
        public readonly int DeploymentSeed;
        int cooldown;
        public sealed class DockLink {public RescueRobot A,B;public int PortA,PortB;public ConfigurableJoint Joint;public LineRenderer Line;}
        Material linkMaterial;

        public RescueSimulation(int sceneIndex,int seed,int nodes=12,RescueDeploymentMode deployment=RescueDeploymentMode.EntranceGrid,int deploymentSeed=907231,int? sourceSeed=null)
        {
            Physics.simulationMode=SimulationMode.Script;Physics.gravity=new Vector3(0,-9.81f,0);Physics.defaultSolverIterations=16;Physics.defaultSolverVelocityIterations=8;
            Scene=new RescueScene(sceneIndex,seed,sourceSeed);linkMaterial=new Material(Shader.Find("Sprites/Default")){color=new Color(.2f,1,.77f)};
            DeploymentMode=deployment;DeploymentSeed=deploymentSeed;var releases=RescueDeployment.Plan(nodes,deploymentSeed,deployment,Scene.ComplexGeometry?Scene.DeploymentRegions:null);
            for(int i=0;i<nodes;i++)
            {
                var pose=releases[i];var robot=new RescueRobot("N"+i.ToString("D2"),pose.Position,seed+i*109);
                if(Scene.ComplexGeometry){robot.AvoidUnsupportedEdges=true;robot.TravelBounds=new Vector4(-.69f,.69f,.01f,2.94f);}
                if(deployment==RescueDeploymentMode.WholeSceneDrop)robot.ConfigureDrop(pose.Rotation,pose.Step);
                Robots.Add(robot);
            }
            Physics.SyncTransforms();
        }
        public void Tick()
        {
            foreach(var robot in Robots)robot.ReleaseAt(Step);
            foreach(var robot in Robots)robot.Motor(Step);
            if(Scene.Spec.Disaster.Contains("洪水")||Scene.Spec.Disaster.Contains("浸水"))foreach(var robot in Robots)if(robot.Core.position.z>1.0f&&robot.Core.position.y<.035f)robot.Core.AddForce(-robot.Core.linearVelocity*.08f);
            if(Docking&&Step>=cooldown&&Step%10==0)TryDock();
            Scene.Aftershock(Step);Physics.Simulate(.02f);Step++;
            foreach(var robot in Robots)robot.AfterPhysics();
            for(int i=Links.Count-1;i>=0;i--)
            {
                var link=Links[i];if(!link.Joint||link.A.Lost||link.B.Lost){Release(i);continue;}
                if(Step%200==0&&(link.A.Stuck||link.B.Stuck)){Release(i);cooldown=Step+150;continue;}
                link.Line.SetPosition(0,link.A.Port(link.PortA));link.Line.SetPosition(1,link.B.Port(link.PortB));
            }
        }
        public void TryDock()
        {
            foreach(var a in Robots)foreach(var b in Robots)
            {
                if(!a.Released||!b.Released||a.Lost||b.Lost||string.CompareOrdinal(a.Id,b.Id)>=0)continue;
                if((a.Core.position-b.Core.position).sqrMagnitude>.08f*.08f)continue;
                if(Connected(a,b))continue;bool linked=false;
                for(int pa=0;pa<RescueRobot.PortCount&&!linked;pa++)for(int pb=0;pb<RescueRobot.PortCount;pb++)
                {
                    if(a.PortBusy[pa]||b.PortBusy[pb])continue;
                    var pointA=a.Port(pa);var pointB=b.Port(pb);float capture=pa<2&&pb<2?.009f:.005f;
                    if(Vector3.Distance(pointA,pointB)>capture||Vector3.Dot(a.PortDirection(pa),b.PortDirection(pb))>-.65f)continue;
                    if((a.PortBody(pa).GetPointVelocity(pointA)-b.PortBody(pb).GetPointVelocity(pointB)).magnitude>.1f)continue;
                    if(Physics.Linecast(pointA,pointB,1<<6))continue;
                    Attach(a,pa,b,pb);linked=true;break;
                }
            }
        }
        bool Connected(RescueRobot start,RescueRobot target)
        {
            var visited=new HashSet<RescueRobot>{start};var queue=new Queue<RescueRobot>();queue.Enqueue(start);
            while(queue.Count>0){var node=queue.Dequeue();if(node==target)return true;foreach(var link in Links){var other=link.A==node?link.B:link.B==node?link.A:null;if(other!=null&&visited.Add(other))queue.Enqueue(other);}}
            return false;
        }
        void Attach(RescueRobot a,int pa,RescueRobot b,int pb)
        {
            var bodyA=a.PortBody(pa);var bodyB=b.PortBody(pb);
            var joint=bodyA.gameObject.AddComponent<ConfigurableJoint>();joint.connectedBody=bodyB;joint.autoConfigureConnectedAnchor=false;
            joint.anchor=bodyA.transform.InverseTransformPoint(a.Port(pa));joint.connectedAnchor=bodyB.transform.InverseTransformPoint(b.Port(pb));
            joint.xMotion=joint.yMotion=joint.zMotion=ConfigurableJointMotion.Locked;joint.angularXMotion=joint.angularYMotion=joint.angularZMotion=ConfigurableJointMotion.Limited;
            joint.lowAngularXLimit=new SoftJointLimit{limit=-45};joint.highAngularXLimit=new SoftJointLimit{limit=45};joint.angularYLimit=joint.angularZLimit=new SoftJointLimit{limit=45};joint.breakForce=.8f;joint.breakTorque=.005f;joint.enableCollision=false;
            var g=new GameObject("Dock_"+a.Id+"_"+b.Id);g.transform.SetParent(Scene.Root.transform,false);var line=g.AddComponent<LineRenderer>();line.sharedMaterial=linkMaterial;line.startWidth=line.endWidth=.0018f;line.positionCount=2;
            a.PortBusy[pa]=true;b.PortBusy[pb]=true;Links.Add(new DockLink{A=a,B=b,PortA=pa,PortB=pb,Joint=joint,Line=line});Attached++;
        }
        void Release(int index)
        {
            var link=Links[index];if(link.Joint)UnityEngine.Object.DestroyImmediate(link.Joint);if(link.Line)UnityEngine.Object.DestroyImmediate(link.Line.gameObject);
            link.A.PortBusy[link.PortA]=false;link.B.PortBusy[link.PortB]=false;Links.RemoveAt(index);Detached++;
        }
        public void ReleaseAll(){for(int i=Links.Count-1;i>=0;i--)Release(i);cooldown=Step+150;}
        public void Power(bool on){foreach(var robot in Robots)robot.Powered=on&&!robot.Lost;}
        public void ApplyGuidance(JObject reply)
        {
            foreach(var robot in Robots)robot.HasTarget=false;
            if(!(reply["goals"] is JArray goals))return;
            foreach(var goal in goals){var robot=Robots.Find(r=>r.Id==(string)goal["node_id"]);if(robot==null)continue;robot.ExplorationTarget=RescueEvaluation.Vector(goal["position"]);robot.HoldUntilStep=(int?)goal["hold_until_step"]??0;robot.HasTarget=true;}
        }
        public RescuePacket Observe(string run,long seq)
        {
            var packet=new RescuePacket{run_id=run,seq_no=seq,sim_step=Step};foreach(var robot in Robots){var scan=robot.Observe(Scene,Step,Robots);if(scan!=null)packet.scans.Add(scan);}return packet;
        }
        public void Dispose(){ReleaseAll();foreach(var robot in Robots)robot.Dispose();Scene.Dispose();if(linkMaterial)UnityEngine.Object.DestroyImmediate(linkMaterial);}
    }

    /// This class is the only consumer of both the inferred map and simulator truth.
    /// Its metrics are never sent to the mapping server or used for action selection.
    public sealed class RescueEvaluation
    {
        public int Observed, PredictedOccupied, CorrectOccupied, TrueOccupied, CueCount, Matched, FalseAlarms;
        public float IoU, GlobalIoU, Precision, Recall, Coverage, PoseRmse, LocalizationError=float.NaN;
        public int TruthOccupiedTotal;
        int observedCorrect,observedPredicted,observedTrue;
        readonly HashSet<Vector3Int> truthGrid=new HashSet<Vector3Int>();
        int truthStep=-1000;
        public int[][] TrainingTruthGrid(){var keys=new List<int[]>();foreach(var key in truthGrid)keys.Add(new[]{key.x,key.y,key.z});return keys.ToArray();}
        public void Evaluate(RescueSimulation simulation,JObject reply)
        {
            Observed=PredictedOccupied=CorrectOccupied=TrueOccupied=Matched=FalseAlarms=0;
            observedCorrect=observedPredicted=observedTrue=0;
            float cell=(float)reply["cell_m"];
            if(simulation.Step-truthStep>=500)
            {
                truthGrid.Clear();
                for(int x=-16;x<16;x++)for(int y=-3;y<18;y++)for(int z=-6;z<64;z++)
                {
                    var center=new Vector3((x+.5f)*cell,(y+.5f)*cell,(z+.5f)*cell);
                    if(Physics.CheckBox(center,Vector3.one*(cell*.42f),Quaternion.identity,1<<6))truthGrid.Add(new Vector3Int(x,y,z));
                }
                truthStep=simulation.Step;
            }
            TruthOccupiedTotal=truthGrid.Count;
            foreach(var c in (JArray)reply["cells"])
            {
                var key=(JArray)c["key"];var p=new Vector3(((int)key[0]+.5f)*cell,((int)key[1]+.5f)*cell,((int)key[2]+.5f)*cell);
                bool truth=truthGrid.Contains(new Vector3Int((int)key[0],(int)key[1],(int)key[2]));bool predicted=(float)c["occupancy"]>.6f;
                if(truth)TrueOccupied++;if(predicted)PredictedOccupied++;if(truth&&predicted)CorrectOccupied++;
                if((int)c["observations"]>0){Observed++;if(truth)observedTrue++;if(predicted)observedPredicted++;if(truth&&predicted)observedCorrect++;}
            }
            Precision=PredictedOccupied>0?(float)CorrectOccupied/PredictedOccupied:0;Recall=TruthOccupiedTotal>0?(float)CorrectOccupied/TruthOccupiedTotal:0;
            int union=observedTrue+observedPredicted-observedCorrect;IoU=union>0?(float)observedCorrect/union:0;Coverage=Observed/(32f*21*70);
            int globalUnion=TruthOccupiedTotal+PredictedOccupied-CorrectOccupied;GlobalIoU=globalUnion>0?(float)CorrectOccupied/globalUnion:0;
            float squared=0;int count=0;foreach(var n in (JArray)reply["nodes"])
            {
                var robot=simulation.Robots.Find(r=>r.Id==(string)n["node_id"]);if(robot==null)continue;var p=Vector(n["estimated_position"]);squared+=(p-robot.Core.position).sqrMagnitude;count++;
            }
            PoseRmse=count>0?Mathf.Sqrt(squared/count):0;
            var used=new HashSet<int>();float error=0;CueCount=((JArray)reply["cues"]).Count;
            foreach(var cue in (JArray)reply["cues"])
            {
                var p=Vector(cue["position"]);int best=-1;float distance=float.PositiveInfinity;
                for(int i=0;i<simulation.Scene.VictimPositions.Count;i++){if(used.Contains(i))continue;float d=Vector3.Distance(p,simulation.Scene.VictimPositions[i]);if(d<distance){distance=d;best=i;}}
                if(best>=0&&distance<.25f){used.Add(best);Matched++;error+=distance;}else FalseAlarms++;
            }
            LocalizationError=Matched>0?error/Matched:float.NaN;
        }
        public static Vector3 Vector(JToken a)=>new Vector3((float)a[0],(float)a[1],(float)a[2]);
        public object Snapshot(RescueSimulation sim)=>new{scene=sim.Scene.Spec.Id,steps=sim.Step,nodes=sim.Robots.Count,released_nodes=sim.Robots.FindAll(r=>r.Released).Count,deployment=sim.DeploymentMode.ToString(),deployment_seed=sim.DeploymentSeed,lost_nodes=sim.Robots.FindAll(r=>r.Lost).Count,observed_cells=Observed,coverage=Coverage,observed_iou=IoU,global_iou=GlobalIoU,truth_occupied_cells=TruthOccupiedTotal,predicted_occupied_cells=PredictedOccupied,correct_occupied_cells=CorrectOccupied,occupancy_precision=Precision,occupancy_recall=Recall,human_precision=CueCount>0?(float?)Matched/CueCount:null,human_recall=sim.Scene.VictimPositions.Count>0?(float?)Matched/sim.Scene.VictimPositions.Count:null,pose_rmse_m=PoseRmse,cues=CueCount,matched=Matched,false_alarms=FalseAlarms,truth_people=sim.Scene.VictimPositions.Count,localization_error_m=float.IsNaN(LocalizationError)?(float?)null:LocalizationError,couplings=sim.Attached,releases=sim.Detached};
    }
}
