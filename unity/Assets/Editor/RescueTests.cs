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
        public static void DashboardSmoke()
        {
            Reset();
            foreach(string name in new[]{"RescueEchoDepth","RescueEchoWire"})
            {var shader=Resources.Load<Shader>(name);if(shader==null||UnityEditor.ShaderUtil.ShaderHasError(shader))throw new Exception("Echo shader failed: "+name);}
            var cells=new JArray();
            for(int x=-2;x<=3;x++)for(int y=-2;y<=3;y++)for(int z=-2;z<=3;z++)
            {
                bool occupied=x>=0&&x<=1&&y>=0&&y<=1&&z>=0&&z<=1;
                cells.Add(new JObject{["key"]=new JArray(x,y,z),["observations"]=0,["sensor_occupancy"]=occupied?.95:.05,["occupancy"]=.99});
            }
            var reply=new JObject{["cell_m"]=.05,["sim_step"]=600,["cells"]=cells};
            using(var view=new RescueEchoWireMap())
            {
                view.Update(reply);if(view.Quads!=0||view.Segments!=0)throw new Exception("Unobserved prior generated a surface");
                foreach(var c in cells)c["observations"]=2;
                view.Update(reply);
                if(view.Quads!=24||view.Segments!=48)throw new Exception($"Surface net must have 24 quads and 48 edges without triangle diagonals: {view.Quads}/{view.Segments}");
                if(view.MaximumRelaxation>.01751f)throw new Exception("Surface relaxation exceeded its display-only bound");
                var baseReply=(JObject)reply.DeepClone();
                var gapReply=(JObject)reply.DeepClone();var missing=((JArray)gapReply["cells"]).First(c=>(int)c["key"][0]==2&&(int)c["key"][1]==0&&(int)c["key"][2]==0);missing.Remove();
                using(var gapView=new RescueEchoWireMap())
                {
                    gapView.Update(gapReply);int incomplete=gapView.Quads;
                    ((JArray)gapReply["cells"]).Add(new JObject{["key"]=new JArray(2,0,0),["observations"]=0,["occupancy"]=.12,["sensor_occupancy"]=JValue.CreateNull(),["source"]="local_surface_completion"});
                    gapView.Update(gapReply);if(gapView.InferredCells!=1||gapView.Quads<=incomplete)throw new Exception("Server inference did not repair a surface gap");
                }
                using(var colorView=new RescueEchoWireMap())
                {
                    var colorReply=(JObject)baseReply.DeepClone();
                    var upper=((JArray)baseReply["cells"]).Select(c=>{var copy=(JObject)c.DeepClone();copy["key"][1]=(int)copy["key"][1]+8;return copy;}).ToArray();
                    foreach(var c in upper)((JArray)colorReply["cells"]).Add(c);
                    colorReply["cues"]=new JArray(new JObject{["position"]=new JArray(.05,.05,.05),["radius_m"]=.2});
                    colorView.Update(colorReply);if(colorView.PersonSegments!=48||colorView.Segments!=96)throw new Exception("Person wire color leaked into a different storey");
                    colorReply["cues"]=new JArray();colorReply["hazards"]=new JArray(new JObject{["position"]=new JArray(.05,.05,.05),["radius_m"]=.2,["kind"]="water"});
                    colorView.Update(colorReply);if(colorView.WaterSegments!=48)throw new Exception("Water wire color missing");
                    colorReply["hazards"][0]["kind"]="gas";colorView.Update(colorReply);if(colorView.GasSegments!=48)throw new Exception("Gas wire color missing");
                }
                var before=view.ReconstructedVertices;
                reply["cues"]=new JArray(new JObject{["position"]=new JArray(0,0,0),["radius_m"]=.2});reply["nodes"]=new JArray(new JObject{["estimated_position"]=new JArray(0,0,0)});reply["hazards"]=new JArray(new JObject{["kind"]="gas",["position"]=new JArray(0,0,0),["radius_m"]=.2});
                foreach(var c in cells)c["occupancy"]=.01;
                view.Update(reply);
                if(!before.SequenceEqual(view.ReconstructedVertices))throw new Exception("Prior or annotation changed the observed surface");
                if(view.Root.transform.childCount!=2||view.Root.GetComponentsInChildren<LineRenderer>().Length!=0)throw new Exception("Decorative overlays remained in the wire-only map");
                foreach(var c in cells)c["sensor_occupancy"]=(float)c["sensor_occupancy"]>.5f?.81:.19;
                view.Update(reply);if(!before.SequenceEqual(view.ReconstructedVertices))throw new Exception("Small sensor probability changes made the wire surface jitter");
                foreach(var c in cells)c["observations"]=1;
                using(var weak=new RescueEchoWireMap()){weak.Update(reply);if(weak.Segments!=0)throw new Exception("Single weak observation was displayed as a measured structure");}
            }
            string output=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/echo-wire-v34-smoke.json"));
            File.WriteAllText(output,JsonConvert.SerializeObject(new{status="pass",checks=new[]{"shader compilation","unobserved and weakly observed structure hidden","connected surface-net quads without triangle diagonals","bounded display relaxation","learned prior does not change observed wire geometry","no point cloud, alert markers or animated overlays","display hysteresis prevents probability jitter","backend completion repairs missing surface quads","person water gas grid colors respect floor height"}},Formatting.Indented));
            Debug.Log("ECHO_WIRE_V34_SMOKE PASS");
        }
        public static void ComplexSmoke()
        {
            Reset();var rows=new List<object>();
            using(var service=new BundledInference())
            {
                service.Start(Path.Combine(Application.streamingAssetsPath,"inference"));
                for(int scene=15;scene<RescueScene.Catalog.Length;scene++)
                {
                    using(var sim=new RescueSimulation(scene,20261031+scene*97,24,RescueDeploymentMode.WholeSceneDrop,907231+scene*13,20261031))
                    using(var client=new TcpClient())
                    {
                        for(int retry=0;retry<100&&!client.Connected;retry++){try{client.Connect("127.0.0.1",service.Port);}catch(SocketException){System.Threading.Thread.Sleep(50);}}
                        if(!client.Connected)throw new Exception("Inference did not start");
                        client.ReceiveTimeout=60000;var stream=client.GetStream();var reader=new StreamReader(stream,Encoding.UTF8);var writer=new StreamWriter(stream,new UTF8Encoding(false)){AutoFlush=true};
                        if(!sim.Scene.ComplexGeometry||sim.Scene.DeploymentRegions.Count<2||sim.Scene.Root.GetComponentsInChildren<Collider>().Any(c=>c.name=="Foundation"))throw new Exception("Scene remained a rectangular tray");
                        foreach(var robot in sim.Robots)if(!sim.Scene.DeploymentRegions.Exists(r=>r.Contains(new Vector2(robot.DeploymentAnchor.x,robot.DeploymentAnchor.z))))throw new Exception("Deployment outside declared envelope");
                        foreach(var p in sim.Scene.VictimPositions)if(Physics.CheckSphere(p,.012f,1<<6))throw new Exception("Person source is inside solid geometry");
                        JObject reply=null;long sequence=0;
                        for(int step=0;step<750;step++)
                        {
                            sim.Tick();if(sim.Step%125!=0)continue;var packet=sim.Observe("complex-"+scene,sequence++);packet.request_map=sim.Step==750;
                            writer.WriteLine(JsonConvert.SerializeObject(packet));reply=JObject.Parse(reader.ReadLine());
                            if((string)reply["type"]=="error")throw new Exception(reply.ToString());sim.ApplyGuidance(reply);
                        }
                        if((string)reply["completion_family"]!="local-plane-v1"||(bool?)reply["distributed_swarm"]!=true)throw new Exception("Complex scene used a legacy geometry template");
                        if(((JArray)reply["cells"]).Count<20)throw new Exception("Complex scene produced no inferred map");
                        using(var map=new RescueEchoWireMap()){map.Update(reply);if(map.Quads==0)throw new Exception("Complex scene produced no wire geometry");}
                        int lost=sim.Robots.Count(r=>r.Lost);if(lost>sim.Robots.Count-4)Debug.Log("LOST_DETAIL "+sim.Scene.Spec.Id+" "+string.Join(";",sim.Robots.Where(r=>r.Lost).Select(r=>r.Id+":"+r.Core.position)));if(lost>sim.Robots.Count-4)throw new Exception("Fewer than four independent observers remain: "+sim.Scene.Spec.Id+"/"+lost);
                        int[] occupiedFloors=sim.Robots.Where(r=>!r.Lost).Select(r=>Mathf.Clamp(Mathf.RoundToInt((r.Core.position.y-.02f)/.28f),0,sim.Scene.FloorCount-1)).Distinct().ToArray();
                        if(sim.Scene.FloorCount>1&&occupiedFloors.Length<2)throw new Exception("Deployment never reached multiple storeys: "+sim.Scene.Spec.Id);
                        var evaluation=new RescueEvaluation();evaluation.Evaluate(sim,reply);
                        rows.Add(new{scene=sim.Scene.Spec.Id,floors=sim.Scene.FloorCount,occupied_floors=occupiedFloors,nodes=24,lost,observed_cells=(int)reply["observed_cells"],completed_cells=((JArray)reply["cells"]).Count(c=>(string)c["source"]=="local_surface_completion"),goals=((JArray)reply["goals"]).Count,global_iou=evaluation.GlobalIoU,coverage=evaluation.Coverage,cues=evaluation.CueCount,matched=evaluation.Matched,false_alarms=evaluation.FalseAlarms,truth_people=sim.Scene.VictimPositions.Count,seconds=15});
                        Debug.Log("COMPLEX_MAP PASS "+sim.Scene.Spec.Id+" floors="+string.Join(",",occupiedFloors)+" lost="+lost);
                    }
                }
            }
            File.WriteAllText(Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/complex-v34-smoke.json")),JsonConvert.SerializeObject(new{status="pass",purpose="functional integration over 15 seconds; loss and accuracy are recorded, not acceptance-certified",accuracy_95_certified=false,cases=rows.Count,source="real simulated range/pose/audio/thermal/environment observations over TCP",truth_to_server=false,rows},Formatting.Indented));
        }
        public static void PortableSmoke()
        {
            Reset();CheckMapReveal();int randomized=0;
            for(int i=0;i<15;i++)
            {
                Vector3[] first;
                using(var scene=new RescueScene(i,20261027,101))
                {first=scene.VictimPositions.ToArray();scene.ShowTruth(true);foreach(var renderer in scene.VictimRenderers)if(!renderer.enabled||renderer.GetComponent<Collider>())throw new Exception("Truth beacon must be visible and have no collider");}
                using(var scene=new RescueScene(i,20261027,202))if(first.Length>0){if(first.SequenceEqual(scene.VictimPositions))throw new Exception("Source randomization did not change positions");randomized++;}
            }
            int firstPid;string folder=Path.Combine(Application.streamingAssetsPath,"inference");
            using(var second=new BundledInference())
            {
                second.Start(folder);
                using(var first=new BundledInference())
                {
                    first.Start(folder);firstPid=first.ProcessId;if(first.Port==second.Port)throw new Exception("App instances share a port");
                    foreach(var service in new[]{first,second})
                    {
                        bool connected=false;for(int retry=0;retry<100;retry++){try{using(var client=new TcpClient("127.0.0.1",service.Port)){connected=true;break;}}catch(SocketException){System.Threading.Thread.Sleep(50);}}
                        if(!connected||!service.Alive)throw new Exception("Bundled server did not start");
                    }
                }
                try{using(var gone=System.Diagnostics.Process.GetProcessById(firstPid))if(!gone.HasExited)throw new Exception("Owned inference leaked after Dispose");}catch(ArgumentException){}
                if(!second.Alive)throw new Exception("Closing one app killed another inference instance");
            }
            string path=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/portable-v32-smoke.json"));Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path,JsonConvert.SerializeObject(new{status="pass",source_randomized_scenes=randomized,surface="only measured faces; adjacent faces culled",inference="two independent ports, child cleanup and isolation verified",truth="red spheres without colliders; not transmitted"},Formatting.Indented));Debug.Log("PORTABLE_V32_SMOKE PASS");
        }
        public static void DockingSmoke()
        {
            Reset();var rows=new List<object>();
            foreach(string type in new[]{"tip-tip","sphere-sphere","tip-sphere"})using(var sim=new RescueSimulation(0,20261028,2))
            {
                PlaceFixture(sim.Robots[0],new Vector3(0,.08f,0),Quaternion.identity);
                PlaceFixture(sim.Robots[1],type=="sphere-sphere"?new Vector3(.022f,.08f,0):new Vector3(0,.08f,type=="tip-tip"?.072f:.044f),type=="tip-sphere"?Quaternion.Euler(0,90,0):Quaternion.identity);
                Physics.SyncTransforms();int testA=type=="sphere-sphere"?2:0,testB=type=="tip-tip"?1:type=="sphere-sphere"?3:2;
                Debug.Log($"DOCK_FIXTURE {type} coreA={sim.Robots[0].Core.position:F4} coreB={sim.Robots[1].Core.position:F4} portA={sim.Robots[0].Port(testA):F4} portB={sim.Robots[1].Port(testB):F4} dot={Vector3.Dot(sim.Robots[0].PortDirection(testA),sim.Robots[1].PortDirection(testB))} blocked={Physics.Linecast(sim.Robots[0].Port(testA),sim.Robots[1].Port(testB),1<<6)}");
                sim.TryDock();if(sim.Links.Count!=1)throw new Exception("Missing or duplicate docking: "+type);
                var link=sim.Links[0];bool sphereA=link.PortA>=2,sphereB=link.PortB>=2;
                if(type=="sphere-sphere"&&(!sphereA||!sphereB)||type=="tip-sphere"&&sphereA==sphereB||type=="tip-tip"&&(sphereA||sphereB))throw new Exception("Wrong dock bodies: "+type);
                if(link.Joint.GetComponent<Rigidbody>()!=link.A.PortBody(link.PortA)||link.Joint.connectedBody!=link.B.PortBody(link.PortB))throw new Exception("Dock joint uses wrong rigid body");
                int pa=link.PortA,pb=link.PortB;sim.TryDock();if(sim.Links.Count!=1)throw new Exception("Duplicate or cyclic dock");
                sim.ReleaseAll();if(sim.Links.Count!=0||sim.Robots.Any(r=>r.ConnectionCount!=0))throw new Exception("Port was not freed");
                rows.Add(new{type,port_a=pa,port_b=pb,status="pass"});
            }
            using(var sim=new RescueSimulation(0,20261028,2))
            {
                PlaceFixture(sim.Robots[0],new Vector3(0,.08f,0),Quaternion.identity);PlaceFixture(sim.Robots[1],new Vector3(.022f,.08f,0),Quaternion.identity);
                var wall=GameObject.CreatePrimitive(PrimitiveType.Cube);wall.layer=6;wall.transform.position=new Vector3(.011f,.08f,0);wall.transform.localScale=new Vector3(.001f,.03f,.03f);Physics.SyncTransforms();sim.TryDock();
                if(sim.Links.Count!=0)throw new Exception("Magnetic connection passed through debris");UnityEngine.Object.DestroyImmediate(wall);rows.Add(new{type="occluded-sphere",status="pass"});
            }
            using(var dry=new RescueScene(0,20261028))if(dry.GasReactionAt(new Vector3(0,.01f,1.5f))!=0||dry.WaterDepthAt(new Vector3(0,.01f,1.5f))!=0)throw new Exception("Dry scene produced a hidden hazard");
            using(var wet=new RescueScene(13,20261028))
            {
                if(!wet.GasLeakPosition.HasValue||wet.GasReactionAt(wet.GasLeakPosition.Value)<.8f)throw new Exception("Gas sensor fixture has no reaction");
                bool found=false;for(float x=-.5f;x<=.5f;x+=.1f)for(float z=1.1f;z<2.4f;z+=.1f)if(wet.WaterDepthAt(new Vector3(x,.012f,z))>.004f)found=true;
                if(!found)throw new Exception("Water sensor fixture has no reaction");
            }
            CheckMapReveal();
            string output=Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/spherical-docking-smoke.json"));Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,JsonConvert.SerializeObject(new{status="pass",ports_per_node=RescueRobot.PortCount,rows,environment_fixture="gas/water/dry verified",progressive_map="unobserved hidden; measured revealed; only nearby completion",truth_positions_to_server=false},Formatting.Indented));Debug.Log("SPHERICAL_DOCKING_SMOKE PASS");
        }
        static void CheckMapReveal()
        {
            using(var view=new RescueMapView())
            {
                var reply=JObject.FromObject(new{cell_m=.05,sim_step=100,cells=new[]{new{key=new[]{0,0,0},occupancy=.9,sensor_occupancy=.9,observations=0},new{key=new[]{1,0,0},occupancy=.9,sensor_occupancy=.1,observations=0},new{key=new[]{10,0,10},occupancy=.9,sensor_occupancy=.1,observations=0}},nodes=new object[0],cues=new object[0]});
                view.Update(reply);if(view.VisibleObservedCells!=0||view.VisibleCompletionCells!=0||view.SurfacePoints!=0||view.SurfaceTriangles!=0)throw new Exception("Unobserved prior appeared in the map");
                reply["cells"][0]["observations"]=2;view.Update(reply);if(view.VisibleObservedCells!=1||view.VisibleCompletionCells!=0||view.SurfacePoints==0)throw new Exception("Observation reveal failed");
                reply["sim_step"]=600;view.Update(reply);if(view.VisibleObservedCells!=1||view.VisibleCompletionCells!=1)throw new Exception("Completion expanded beyond the observed neighborhood");
                if(view.SurfaceTriangles!=12)throw new Exception("An unobserved prior acquired solid geometry");
                reply["cells"][1]["sensor_occupancy"]=.9;reply["cells"][1]["observations"]=2;view.Update(reply);if(view.SurfaceTriangles!=20)throw new Exception("Internal shared voxel face was not removed");
                view.SetSurfaceMode(false);if(view.SurfaceTriangles!=20)throw new Exception("View mode mutated reconstruction");
            }
        }
        static void PlaceFixture(RescueRobot robot,Vector3 target,Quaternion rotation)
        {
            var origin=robot.Core.position;foreach(var body in new[]{robot.Core,robot.Cones[0],robot.Cones[1]}){var p=target+rotation*(body.position-origin);var q=rotation*body.rotation;body.transform.SetPositionAndRotation(p,q);body.position=p;body.rotation=q;body.linearVelocity=Vector3.zero;body.angularVelocity=Vector3.zero;}
        }
        public static void DistributedSmoke()
        {
            Reset();var results=new List<object>();
            for(int scene=0;scene<15;scene++)
            {
                Vector3[] reference=null;
                foreach(int count in new[]{24,36,48})
                using(var sim=new RescueSimulation(scene,20261027+scene*997,count,RescueDeploymentMode.WholeSceneDrop,907231+scene*13))
                {
                    var sources=sim.Scene.VictimPositions.ToArray();
                    if(reference==null)reference=sources;else if(!reference.SequenceEqual(sources))throw new Exception("Changing node count changed victim placement");
                    if(sim.Robots.Count!=count||sim.Robots.Any(r=>r.Released||!r.Core.isKinematic))throw new Exception("Incorrect drop waiting state");
                    float minX=sim.Robots.Min(r=>r.DeploymentAnchor.x),maxX=sim.Robots.Max(r=>r.DeploymentAnchor.x),minZ=sim.Robots.Min(r=>r.DeploymentAnchor.z),maxZ=sim.Robots.Max(r=>r.DeploymentAnchor.z);
                    if(maxX-minX<.8f||maxZ-minZ<2.0f)throw new Exception("Release plan is not distributed across the scene");
                    foreach(var a in sim.Robots)foreach(var b in sim.Robots)if(a!=b&&Vector3.Distance(a.DeploymentAnchor,b.DeploymentAnchor)<.0849f)throw new Exception("Release bodies overlap");
                    for(int step=0;step<800;step++)
                    {
                        sim.Tick();if(sim.Step%25!=0)continue;var packet=sim.Observe("distributed-smoke",sim.Step/25-1);
                        foreach(var scan in packet.scans)if(!sim.Robots.Any(r=>r.Id==scan.node_id&&r.Released&&!r.Lost))throw new Exception("An unreleased or lost node transmitted");
                    }
                    if(sim.Robots.Any(r=>!r.Released||!float.IsFinite(r.Core.position.x)||!float.IsFinite(r.Core.position.y)||!float.IsFinite(r.Core.position.z)))throw new Exception("Invalid final deployment state");
                    if(sim.Robots.Any(r=>r.Core.position.y>RescueDeployment.DropHeight-.03f))throw new Exception("A released node did not fall");
                    sim.Power(false);sim.Docking=false;sim.ReleaseAll();for(int settle=0;settle<150;settle++)sim.Tick();
                    var before=sim.Robots.Select(r=>r.Core.position).ToArray();
                    foreach(bool transparent in new[]{false,true})
                    {
                        sim.Scene.Cutaway(transparent);
                        foreach(var collider in sim.Scene.Root.GetComponentsInChildren<Collider>())
                        {
                            var renderer=collider.GetComponent<Renderer>();
                            if(!renderer||!renderer.enabled||!renderer.gameObject.activeInHierarchy)throw new Exception("Invisible collision support: "+collider.name);
                        }
                    }
                    if(!before.SequenceEqual(sim.Robots.Select(r=>r.Core.position)))throw new Exception("View mode moved a robot");
                    float maximumSupportGap=0,maximumNearestSurface=0;int elevated=0,lateralSupports=0;
                    var sceneColliders=sim.Scene.Root.GetComponentsInChildren<Collider>();
                    foreach(var robot in sim.Robots)
                    {
                        if(robot.Lost)continue;float gap=float.PositiveInfinity,nearest=float.PositiveInfinity;
                        foreach(var body in new[]{robot.Core,robot.Cones[0],robot.Cones[1]})
                        {
                            if(body.isKinematic||!body.useGravity)throw new Exception("Released body has no active gravity");
                            if(Physics.Raycast(body.worldCenterOfMass,Vector3.down,out RaycastHit support,.5f,1<<6))gap=Mathf.Min(gap,support.distance);
                            foreach(var collider in sceneColliders)nearest=Mathf.Min(nearest,Vector3.Distance(body.worldCenterOfMass,collider.ClosestPoint(body.worldCenterOfMass)));
                        }
                        // A wedged body can be supported laterally at a slab edge,
                        // so a downward ray alone cannot identify every real contact.
                        if(nearest>.04f)throw new Exception($"Unsupported settled node {sim.Scene.Spec.Id}/{robot.Id}: nearest surface={nearest}, downward={gap}");
                        maximumNearestSurface=Mathf.Max(maximumNearestSurface,nearest);if(float.IsFinite(gap))maximumSupportGap=Mathf.Max(maximumSupportGap,gap);if(gap>.08f)lateralSupports++;if(robot.Core.position.y>.09f)elevated++;
                    }
                    results.Add(new{scene=sim.Scene.Spec.Id,nodes=count,released=sim.Robots.Count(r=>r.Released),lost=sim.Robots.Count(r=>r.Lost),span_x=maxX-minX,span_z=maxZ-minZ,couplings=sim.Attached,detachments=sim.Detached,geometry_colliders=sceneColliders.Length,settled_max_downward_surface_gap_m=maximumSupportGap,settled_max_nearest_surface_distance_m=maximumNearestSurface,lateral_or_edge_supports=lateralSupports,elevated_on_visible_geometry=elevated,invisible_collision_surfaces=0});
                    Debug.Log($"DISTRIBUTED_SMOKE PASS {sim.Scene.Spec.Id} nodes={count} lost={sim.Robots.Count(r=>r.Lost)}");
                }
            }
            string output=Arg("-purumoguOutput",Path.GetFullPath(Path.Combine(Application.dataPath,"../../reports/rescue/distributed-deployment-smoke.json")));Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,JsonConvert.SerializeObject(new{status="pass",cases=results.Count,drop_height_m=RescueDeployment.DropHeight,scatter_seconds=RescueDeployment.ScatterSteps*.02,rows=results},Formatting.Indented));
        }
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
        public static void Evaluate()=>EvaluateImpl(false);
        public static void EvaluateSwarm()=>EvaluateImpl(true);
        static void EvaluateImpl(bool swarm)
        {
            Reset();int port=int.Parse(Arg("-purumoguPort","47120")),steps=int.Parse(Arg("-purumoguSteps","4500")),repeats=int.Parse(Arg("-purumoguRepeats","2")),seedBase=int.Parse(Arg("-purumoguSeedBase","20261027"));string split=Arg("-purumoguSplit","development");bool guidance=Arg("-purumoguGuidance","false")=="true",diagnostics=Arg("-purumoguDiagnostics","false")=="true";
            string path=Path.GetFullPath(Arg("-purumoguReport",Path.Combine(Application.dataPath,"../../reports/rescue/"+split)));Directory.CreateDirectory(path);string raw=Path.GetFullPath(Arg("-purumoguRaw",Path.Combine(Application.dataPath,"../../experiments/raw/rescue")));Directory.CreateDirectory(raw);
            int sceneFilter=int.Parse(Arg("-purumoguSceneFilter","-1"));
            int countFilter=int.Parse(Arg("-purumoguNodeCount","0")),deploymentBase=int.Parse(Arg("-purumoguDeploymentSeedBase","30800000"));
            int[] counts=swarm?(countFilter==0?new[]{24,36,48}:new[]{countFilter}):new[]{12};
            if(swarm&&counts.Any(n=>n!=24&&n!=36&&n!=48))throw new ArgumentException("Swarm count must be 24, 36 or 48");
            string generation=swarm?Arg("-purumoguGeneration","rescue-3.1"):"rescue-2.5",controllerRevision=swarm?"whole-scene-spherical-env-v31":"progress-3cm-v2";
            if(swarm&&generation!="rescue-3.1")throw new ArgumentException("This physical revision requires generation rescue-3.1");
            int expectedMaps=(sceneFilter<0?15:1)*repeats*counts.Length;
            var outcomes=new List<object>();
            File.WriteAllText(Path.Combine(path,"manifest.json"),JsonConvert.SerializeObject(new{status="running",generation,controller_revision=controllerRevision,expected_maps=expectedMaps,steps,seed_base=seedBase,split,guidance,node_counts=counts,deployment_seed_base=deploymentBase},Formatting.Indented));
            using(var training=new StreamWriter(Path.Combine(raw,"features-"+split+".jsonl"),false,new UTF8Encoding(false)))
            using(var grids=new StreamWriter(Path.Combine(raw,"grids-"+split+".jsonl"),false,new UTF8Encoding(false)))
            {
            for(int scene=0;scene<15;scene++)for(int repeat=0;repeat<repeats;repeat++)foreach(int nodes in counts)
            {
                if(sceneFilter>=0&&scene!=sceneFilter)continue;
                int seed=seedBase+repeat*997,deploymentSeed=deploymentBase+scene*10007+repeat*997+nodes*131;
                string groupId="rescue-world-"+RescueScene.Catalog[scene].Id+"-"+seed;
                string run=Guid.NewGuid().ToString("N"),mapId=generation+"-"+RescueScene.Catalog[scene].Id+"-"+seed+(swarm?"-N"+nodes+"-D"+deploymentSeed:"");long seq=0;
                using(var socket=new TcpClient("127.0.0.1",port))using(var sim=new RescueSimulation(scene,seed,nodes,swarm?RescueDeploymentMode.WholeSceneDrop:RescueDeploymentMode.EntranceGrid,deploymentSeed))
                using(var audit=new StreamWriter(Path.Combine(raw,run+".jsonl"),false,new UTF8Encoding(false)))
                {
                    socket.NoDelay=true;socket.ReceiveTimeout=30000;socket.SendTimeout=30000;var stream=socket.GetStream();var reader=new StreamReader(stream,Encoding.UTF8);var writer=new StreamWriter(stream,new UTF8Encoding(false)){AutoFlush=true};var evaluation=new RescueEvaluation();
                    JObject finalMap=null;
                    for(int step=0;step<steps;step++)
                    {
                        sim.Tick();if(sim.Step%25!=0)continue;
                        var packet=sim.Observe(run,seq++);packet.request_map=sim.Step+25>steps;packet.request_diagnostics=diagnostics;string json=JsonConvert.SerializeObject(packet);audit.WriteLine(json);writer.WriteLine(json);string line=reader.ReadLine();if(line==null||line.Length>8388608)throw new IOException("bad map response");var reply=JObject.Parse(line);if((string)reply["type"]=="error")throw new IOException(reply.ToString());if((long)reply["seq_no"]!=packet.seq_no)throw new IOException("sequence mismatch");if(sim.Step==25||sim.Step+25>steps)audit.WriteLine(line);if(guidance)sim.ApplyGuidance(reply);if(packet.request_map)evaluation.Evaluate(sim,reply);
                        finalMap=reply;
                        foreach(var scan in packet.scans){var robot=sim.Robots.Find(r=>r.Id==scan.node_id);training.WriteLine(JsonConvert.SerializeObject(new{kind="acoustic",map_id=mapId,world_group_id=groupId,nodes,deployment_seed=deploymentSeed,split,features=new[]{scan.acoustic_snr_db/20,scan.voice_band_ratio,scan.modulation,scan.temperature_delta_c/2},label=robot.EvaluationSourceIsHuman?1:0}));}
                    }
                    if(finalMap!=null)foreach(var c in (JArray)finalMap["cells"])
                    {
                        var k=(JArray)c["key"];var p=new Vector3(((int)k[0]+.5f)*RescueScene.Cell,((int)k[1]+.5f)*RescueScene.Cell,((int)k[2]+.5f)*RescueScene.Cell);bool truth=Physics.CheckBox(p,Vector3.one*(RescueScene.Cell*.42f),Quaternion.identity,1<<6);float prob=(float)c["occupancy"];training.WriteLine(JsonConvert.SerializeObject(new{kind="occupancy",map_id=mapId,split,features=new[]{Mathf.Log(prob/(1-prob)),Mathf.Log(1+(int)c["observations"])},label=truth?1:0}));
                    }
                    var row=JObject.FromObject(evaluation.Snapshot(sim));row["seed"]=seed;row["repeat"]=repeat;row["map_id"]=mapId;row["world_group_id"]=groupId;row["run_id"]=run;row["split"]=split;outcomes.Add(row);File.WriteAllText(Path.Combine(path,"outcomes.json"),JsonConvert.SerializeObject(outcomes,Formatting.Indented));training.Flush();
                    grids.WriteLine(JsonConvert.SerializeObject(new{scene=sim.Scene.Spec.Id,map_id=mapId,world_group_id=groupId,nodes,deployment_seed=deploymentSeed,run_id=run,split,occupied_keys=evaluation.TrainingTruthGrid(),victim_positions=sim.Scene.VictimPositions.Select(RescueRobot.Array).ToArray()}));grids.Flush();
                    Debug.Log($"RESCUE_EVAL {sim.Scene.Spec.Id} nodes={nodes} repeat={repeat} IoU={evaluation.IoU:F3} coverage={evaluation.Coverage:F3} targets={evaluation.Matched}/{sim.Scene.Spec.Victims} false={evaluation.FalseAlarms}");
                }
            }
            }
            File.WriteAllText(Path.Combine(path,"manifest.json"),JsonConvert.SerializeObject(new{status="complete",generation,controller_revision=controllerRevision,expected_maps=expectedMaps,unity=Application.unityVersion,scenes=sceneFilter<0?15:1,repeats,steps,seconds=steps*.02,node_counts=counts,seed_base=seedBase,deployment_seed_base=deploymentBase,deployment=swarm?"WholeSceneDrop":"EntranceGrid",split,guidance,robot_length_m=.07,robot_mass_kg=.024,cell_m=RescueScene.Cell,inference="Bayesian occupancy / independent-node acoustic bearing fusion",source_placement="independent-position-v1",acoustic_power_db="10log10; all-source power mixture; no human label in observations",truth_to_server=false,targets=new{global_occupancy_iou=.95,human_precision=.95,human_recall=.95}},Formatting.Indented));
        }
    }
}
