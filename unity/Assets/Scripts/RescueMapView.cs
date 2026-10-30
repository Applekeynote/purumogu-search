using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace Purumogu.Rescue
{
    /// Fine observed endpoints overlay a faint 5cm evidence scaffold; neither is ground truth.
    public sealed class RescueMapView : System.IDisposable
    {
        public static readonly Color PersonColor=new Color(1,.70f,.18f),GasColor=new Color(1,.30f,.48f),WaterColor=new Color(1,.40f,.32f);
        public sealed class AlertMark {public string Kind,Id,Label,Reason,Access;public Vector3 Position;public float Radius,Score,Age;public int Observers;public Color Color;}
        public readonly List<AlertMark> Alerts=new List<AlertMark>();
        public readonly List<Vector3> CuePositions=new List<Vector3>();
        public readonly List<string> CueLabels=new List<string>();
        public readonly GameObject Root=new GameObject("InferredMapOnly");
        public readonly RescueMapKnowledge Knowledge=new RescueMapKnowledge();
        public bool ShowCompletion{get;private set;}=true;
        public bool ShowPeople{get;private set;}=true;
        public bool ShowEcho{get;private set;}=true;
        public RescueMapProjection Projection{get;private set;}
        public float SlicePosition{get;private set;}
        readonly Mesh sectionGuide=new Mesh{name="SectionPlaneLocation"};
        readonly Mesh context=new Mesh{name="ObservedFreeVolumeFootprint",indexFormat=IndexFormat.UInt32},inferred=new Mesh{name="DashedEstimatedStructure",indexFormat=IndexFormat.UInt32},echo=new Mesh{name="ObservationArrivalSchematic"};
        readonly List<Vector3> pulseOrigins=new List<Vector3>();
        readonly HashSet<Vector3Int> solidEvidence=new HashSet<Vector3Int>();
        float pulseAt=-10;
        long pulseStep=-1;
        readonly Mesh points=new Mesh{name="ObservedVoxelSurfacePoints",indexFormat=IndexFormat.UInt32};
        readonly Mesh surfaces=new Mesh{name="MeasuredSurfaceFaces",indexFormat=IndexFormat.UInt32},edges=new Mesh{name="MeasuredSilhouetteEdges",indexFormat=IndexFormat.UInt32};
        readonly Material surfaceMaterial;
        readonly Renderer surfaceRenderer,edgeRenderer;
        JObject cachedReply;
        public bool RenderSurfaces{get;private set;}=true;
        public int SurfaceTriangles=>surfaces.triangles.Length/3;
        public int SurfaceGridVertices=>surfaces.vertexCount;
        public float MaximumDisplayShift{get;private set;}
        public const float LatticeSpacingM=.10f;
        readonly Mesh grid=new Mesh{name="QuietCoordinateGrid"},trail=new Mesh{name="EstimatedSelectedNodeTrail"};
        readonly Material pointMaterial,lineMaterial;
        readonly List<Vector3> vertices=new List<Vector3>(),uv=new List<Vector3>();
        readonly List<Color> colors=new List<Color>();
        readonly List<int> indices=new List<int>();
        readonly List<GameObject> markers=new List<GameObject>();
        readonly Dictionary<Vector3Int,float> firstEvidenceTime=new Dictionary<Vector3Int,float>();
        readonly Dictionary<Vector3Int,long> firstEvidence=new Dictionary<Vector3Int,long>();
        readonly Dictionary<string,List<Vector3>> trajectories=new Dictionary<string,List<Vector3>>();
        static readonly Vector2[] Corners={new Vector2(-1,-1),new Vector2(-1,1),new Vector2(1,1),new Vector2(1,-1)};
        string selected="N00";
        public int VisibleObservedCells{get;private set;}
        public int VisibleCompletionCells{get;private set;}
        public int GasZones{get;private set;}
        public int WaterZones{get;private set;}
        public int GasObservedNodes{get;private set;}
        public int WaterObservedNodes{get;private set;}
        public int SurfacePoints{get;private set;}
        public int FineObservedPoints{get;private set;}
        public RescueMapView()
        {
            pointMaterial=new Material(Resources.Load<Shader>("RescueMapPoint"));lineMaterial=new Material(Shader.Find("Sprites/Default"));
            surfaceMaterial=new Material(Resources.Load<Shader>("RescueMapSurface"));
            surfaceRenderer=MeshObject("ObservedFilledFaces",surfaces,surfaceMaterial);edgeRenderer=MeshObject("ObservedObjectEdges",edges,lineMaterial);
            MeshObject("SliceGuide",sectionGuide,lineMaterial);MeshObject("ObservedFreeVolume",context,lineMaterial);MeshObject("EstimatedDashedContours",inferred,lineMaterial);MeshObject("SchematicEchoWaves",echo,lineMaterial);MeshObject("ObservedSurfacePoints",points,pointMaterial);MeshObject("MapGrid",grid,lineMaterial);MeshObject("NodeEstimatedTrail",trail,lineMaterial);points.MarkDynamic();trail.MarkDynamic();
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();var faint=new Color(.19f,.25f,.28f,.26f);
            for(int x=-8;x<=8;x+=2)Line(v,c,ix,new Vector3(x*.1f,-.035f,-.3f),new Vector3(x*.1f,-.035f,3.2f),faint);
            for(int z=-3;z<=32;z+=2)Line(v,c,ix,new Vector3(-.8f,-.035f,z*.1f),new Vector3(.8f,-.035f,z*.1f),faint);
            grid.SetVertices(v);grid.SetColors(c);grid.SetIndices(ix,MeshTopology.Lines,0);grid.RecalculateBounds();
        }
        Renderer MeshObject(string name,Mesh mesh,Material material)
        {var g=new GameObject(name);g.layer=8;g.transform.SetParent(Root.transform,false);g.AddComponent<MeshFilter>().sharedMesh=mesh;var renderer=g.AddComponent<MeshRenderer>();renderer.sharedMaterial=material;return renderer;}
        public void SetSurfaceMode(bool enabled)
        {RenderSurfaces=enabled;surfaceRenderer.enabled=edgeRenderer.enabled=enabled;if(cachedReply!=null)Update(cachedReply);}
        public void SetLayers(bool completion,bool people,bool echoes)
        {if(ShowCompletion==completion&&ShowPeople==people&&ShowEcho==echoes)return;ShowCompletion=completion;ShowPeople=people;ShowEcho=echoes;if(cachedReply!=null)Update(cachedReply);}
        public void SetSection(RescueMapProjection mode,float position)
        {if(Projection==mode&&Mathf.Abs(SlicePosition-position)<.001f)return;Projection=mode;SlicePosition=position;if(cachedReply!=null)Update(cachedReply);}
        bool InSlice(Vector3 p)=>Knowledge.InSlice(p,Projection,SlicePosition);
        void BuildContext()
        {
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();
            var iv=new List<Vector3>();var ic=new List<Color>();var ii=new List<int>();
            var floor=new Dictionary<Vector2Int,Color>();float cell=Knowledge.CellSize;
            foreach(var sample in Knowledge.Cells.Values)
            {
                var p=Knowledge.Center(sample.Key);if(!InSlice(p))continue;
                if(sample.Tier==RescueEvidenceTier.Measured&&!sample.Solid)
                {
                    var key=new Vector2Int(sample.Key.x,sample.Key.z);floor[key]=new Color(.12f,.31f,.34f,.20f);
                }
                if(!ShowCompletion||sample.Tier==RescueEvidenceTier.Unknown||sample.Tier==RescueEvidenceTier.Measured)continue;
                if(!sample.Solid)continue;
                bool high=sample.Tier==RescueEvidenceTier.HighEstimate;
                var color=high?new Color(.43f,.75f,.8f,.30f):new Color(.33f,.45f,.51f,.12f);
                bool occupied(Vector3Int key)=>Knowledge.Cells.TryGetValue(key,out var item)&&item.Tier!=RescueEvidenceTier.Unknown&&item.Solid;
                for(int axis=0;axis<3;axis++)for(int sign=-1;sign<=1;sign+=2)
                {
                    var outside=sample.Key;outside[axis]+=sign;if(occupied(outside))continue;
                    int a=(axis+1)%3,b=(axis+2)%3;var normal=Vector3.zero;normal[axis]=sign;
                    for(int edge=0;edge<4;edge++)
                    {
                        var start=p+normal*cell*.5f;var end=start;var c0=Corners[edge];var c1=Corners[(edge+1)%4];
                        start[a]+=c0.x*cell*.5f;start[b]+=c0.y*cell*.5f;end[a]+=c1.x*cell*.5f;end[b]+=c1.y*cell*.5f;
                        var tangent=(start+end)*.5f-(p+normal*cell*.5f);int along=Mathf.Abs(tangent[a])>Mathf.Abs(tangent[b])?a:b;
                        var adjacent=sample.Key;adjacent[along]+=tangent[along]>0?1:-1;var adjacentOutside=adjacent;adjacentOutside[axis]+=sign;
                        if(occupied(adjacent)&&!occupied(adjacentOutside))continue;
                        Line(iv,ic,ii,start,Vector3.Lerp(start,end,high?.55f:.18f),color);
                    }
                }
            }
            // A quiet footprint shows measured void versus the unlit, unknown space.
            if(Projection==RescueMapProjection.Overview||Projection==RescueMapProjection.Top)
            foreach(var item in floor)
            {
                var p=new Vector3((item.Key.x+.5f)*cell,-.151f,(item.Key.y+.5f)*cell);int n=v.Count;
                foreach(var corner in Corners){v.Add(p+new Vector3(corner.x,0,corner.y)*cell*.48f);c.Add(item.Value);}
                ix.AddRange(new[]{n,n+1,n+2,n,n+2,n+3});
            }
            var gv=new List<Vector3>();var gc=new List<Color>();var gi=new List<int>();
            if(Projection==RescueMapProjection.Overview)
            {
                var a=new Vector3(SlicePosition,-.15f,-.3f);var b=new Vector3(SlicePosition,-.15f,3.2f);var color=new Color(.67f,.87f,.9f,.30f);
                for(int i=0;i<35;i++)Line(gv,gc,gi,Vector3.Lerp(a,b,i/35f),Vector3.Lerp(a,b,(i+.55f)/35f),color);
                Line(gv,gc,gi,a,a+Vector3.up*1.05f,color);Line(gv,gc,gi,b,b+Vector3.up*1.05f,color);
            }
            sectionGuide.Clear();sectionGuide.SetVertices(gv);sectionGuide.SetColors(gc);sectionGuide.SetIndices(gi,MeshTopology.Lines,0);sectionGuide.RecalculateBounds();
            context.Clear();context.SetVertices(v);context.SetColors(c);context.SetTriangles(ix,0);context.RecalculateBounds();
            inferred.Clear();inferred.SetVertices(iv);inferred.SetColors(ic);inferred.SetIndices(ii,MeshTopology.Lines,0);inferred.RecalculateBounds();
        }
        public void AnimateEcho()
        {
            surfaceMaterial.SetFloat("_MapNow",Time.realtimeSinceStartup);surfaceMaterial.SetFloat("_EchoEnabled",ShowEcho?1:0);
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();float age=Time.realtimeSinceStartup-pulseAt;
            if(ShowEcho&&age<1.65f)foreach(var origin in pulseOrigins)for(int tail=0;tail<2;tail++)
            {
                float radius=(age-tail*.17f)*.45f;if(radius<0||radius>.75f)continue;
                Vector3 previous=Vector3.zero;bool valid=false;
                for(int i=0;i<=80;i++)
                {
                    float angle=i*Mathf.PI*2/80;var direction=new Vector3(Mathf.Cos(angle),0,Mathf.Sin(angle));float reach=radius;
                    for(float d=.025f;d<radius;d+=.025f)
                    {var q=origin+direction*d;var k=Vector3Int.FloorToInt(q/Knowledge.CellSize);if(solidEvidence.Contains(k)){reach=d;break;}}
                    var p=origin+direction*reach;if(valid&&InSlice(p))Line(v,c,ix,previous,p,new Color(.68f,.96f,1,(1-age/1.65f)*(tail==0?.5f:.15f)));previous=p;valid=true;
                }
            }
            echo.Clear();echo.SetVertices(v);echo.SetColors(c);echo.SetIndices(ix,MeshTopology.Lines,0);echo.RecalculateBounds();
        }
        void BuildSurfaces(HashSet<Vector3Int> supported,float cell)
        {
            var lattice=new RescueSurfaceLattice(supported,cell,p=>Knowledge.Cells.TryGetValue(Vector3Int.FloorToInt(p/cell),out var evidence)&&evidence.Tier==RescueEvidenceTier.Measured);
            MaximumDisplayShift=lattice.MaximumShift;
            var v=new List<Vector3>();var n=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();var gridUV=new List<Vector3>();
            var ev=new List<Vector3>();var ec=new List<Color>();var ei=new List<int>();var emitted=new HashSet<(Vector3Int,Vector3Int)>();
            foreach(var face in lattice.Faces)
            {
                var k=face.Cell;int axis=face.Axis,sign=face.Sign,a=(axis+1)%3,b=(axis+2)%3;
                var center=Knowledge.Center(k);var normal=Vector3.zero;normal[axis]=sign;int start=v.Count;
                firstEvidenceTime.TryGetValue(k,out float arrival);
                foreach(var corner in face.Corners)
                {
                    v.Add(lattice.Positions[corner]);n.Add(lattice.Normals[corner]);c.Add(ObservedColor(center));
                    // Material coordinates travel with the measured surface, bending the grid over its shape.
                    gridUV.Add(new Vector3(corner[a]*cell/LatticeSpacingM,corner[b]*cell/LatticeSpacingM,arrival));
                }
                ix.AddRange(new[]{start,start+1,start+2,start,start+2,start+3});
                // Thin silhouette/crease support. The surface grid is generated without triangle diagonals.
                for(int edge=0;edge<4;edge++)
                {
                    int next=(edge+1)%4;var from=face.Corners[edge];var to=face.Corners[next];
                    var midpoint=((Vector3)from+(Vector3)to)*cell*.5f;var tangent=midpoint-(center+normal*cell*.5f);
                    int along=Mathf.Abs(tangent[a])>Mathf.Abs(tangent[b])?a:b;var adjacent=k;adjacent[along]+=tangent[along]>0?1:-1;
                    var adjacentOutside=adjacent;adjacentOutside[axis]+=sign;if(supported.Contains(adjacent)&&!supported.Contains(adjacentOutside))continue;
                    bool reverse=from.x>to.x||from.x==to.x&&(from.y>to.y||from.y==to.y&&from.z>to.z);
                    if(!emitted.Add(reverse?(to,from):(from,to)))continue;
                    Line(ev,ec,ei,lattice.Positions[from]+lattice.Normals[from]*.0008f,lattice.Positions[to]+lattice.Normals[to]*.0008f,new Color(.28f,.72f,.62f,.38f));
                }
            }
            surfaces.Clear();surfaces.SetVertices(v);surfaces.SetNormals(n);surfaces.SetColors(c);surfaces.SetUVs(0,gridUV);surfaces.SetTriangles(ix,0);surfaces.RecalculateBounds();
            edges.Clear();edges.SetVertices(ev);edges.SetColors(ec);edges.SetIndices(ei,MeshTopology.Lines,0);edges.RecalculateBounds();
        }
        public void SetViewport(float width,float height,float zoom=1){pointMaterial.SetVector("_ViewportSize",new Vector4(Mathf.Max(1,width),Mathf.Max(1,height),0,0));pointMaterial.SetFloat("_PointScale",Mathf.Clamp(zoom,.65f,2.6f));}
        public void SetSelectedNode(string id){if(selected==id)return;selected=id;BuildTrail();}
        void Point(Vector3 center,Color color,float size)
        {
            int n=vertices.Count;foreach(var corner in Corners){vertices.Add(center);colors.Add(color);uv.Add(new Vector3(corner.x,corner.y,size));}
            indices.Add(n);indices.Add(n+1);indices.Add(n+2);indices.Add(n);indices.Add(n+2);indices.Add(n+3);
        }
        static void Line(List<Vector3> v,List<Color> c,List<int> ix,Vector3 a,Vector3 b,Color color)
        {int n=v.Count;v.Add(a);v.Add(b);c.Add(color);c.Add(color);ix.Add(n);ix.Add(n+1);}
        static Vector3Int Key(JToken c){var k=c["key"];return new Vector3Int((int)k[0],(int)k[1],(int)k[2]);}
        Color ObservedColor(Vector3 center)
        {
            var color=Color.Lerp(new Color(.19f,.43f,.47f,.94f),new Color(.48f,.70f,.72f,.98f),Mathf.Clamp01(center.y/.7f));
            foreach(var alert in Alerts)if(alert.Kind!="person"&&alert.Kind!="blockage"&&Mathf.Abs(center.y-alert.Position.y)<.17f&&new Vector2(center.x-alert.Position.x,center.z-alert.Position.z).magnitude<alert.Radius){color=alert.Color;color.a=.95f;if(alert.Kind=="gas")break;}
            return color;
        }
        public void Update(JObject reply)
        {
            cachedReply=reply;Knowledge.Update(reply);
            float cell=(float)reply["cell_m"];long step=(long)reply["sim_step"];
            foreach(var marker in markers)if(marker)Object.DestroyImmediate(marker);markers.Clear();Alerts.Clear();CueLabels.Clear();CuePositions.Clear();
            GasZones=WaterZones=0;GasObservedNodes=(int?)reply["gas_observed_nodes"]??0;WaterObservedNodes=(int?)reply["water_observed_nodes"]??0;
            int number=0;
            foreach(var cue in (JArray)reply["cues"])
            {
                var p=RescueEvaluation.Vector(cue["position"]);float radius=(float)cue["radius_m"];string id=((char)('A'+number++)).ToString();
                var label=$"優先確認 {id} / 人候補・未確認";
                string sound=cue["evidence"]?["human"]!=null?$"音声特徴 {((float)cue["evidence"]["human"]>=.65f?"高":(float)cue["evidence"]["human"]>=.35f?"中":"弱")}":"音声特徴 未取得";
                string heat=cue["evidence"]?["heat"]!=null?$"温度差 {(float)cue["evidence"]["heat"]:+0.00;-0.00;0.00}℃":"温度差 未取得";
                CuePositions.Add(p);CueLabels.Add(label);Alerts.Add(new AlertMark{Kind="person",Id=id,Label=label,Position=p,Radius=radius,Color=PersonColor,Score=(float)cue["score"],Observers=(int)cue["observers"],Age=Mathf.Max(0,step-((long?)cue["last_seen_step"]??step))*.02f,
                    Reason=sound+" / "+heat,Access=(Knowledge.HasObservedCavity(p)?"周辺に観測空洞":"周辺空洞 未確認")+" / "+(Knowledge.HasApproachGoal(reply,p)?"接近目標あり":"到達性 未確認")});
            }
            if(reply["hazards"] is JArray hazards)foreach(var hazard in hazards)
            {
                bool gas=(string)hazard["kind"]=="gas";string id=gas?"G"+(++GasZones):"W"+(++WaterZones);float value=(float)hazard["value"];int observers=(int)hazard["observers"];
                Alerts.Add(new AlertMark{Kind=gas?"gas":"water",Id=id,Position=RescueEvaluation.Vector(hazard["position"]),Radius=(float)hazard["radius_m"],Color=gas?GasColor:WaterColor,
                    Label=gas?$"ガス反応 {id}  ·  相対値 {value:F2}  ·  {observers}ノード":$"浸水 {id}  ·  水深 {value*100:F1}cm  ·  {observers}ノード"});
            }
            var directions=new[]{Vector3Int.left,Vector3Int.right,new Vector3Int(0,0,1),new Vector3Int(0,0,-1)};
            foreach(var c in Knowledge.Cells.Values)
            {
                if(c.Tier!=RescueEvidenceTier.Measured||c.Solid)continue;int walls=0;
                foreach(var direction in directions)if(Knowledge.Cells.TryGetValue(c.Key+direction,out var n)&&n.Tier==RescueEvidenceTier.Measured&&n.Solid)walls++;
                if(walls<4)continue;var p=Knowledge.Center(c.Key);if(Alerts.Exists(a=>a.Kind=="blockage"&&Vector3.Distance(a.Position,p)<.35f))continue;
                Alerts.Add(new AlertMark{Kind="blockage",Id="B"+(Alerts.FindAll(a=>a.Kind=="blockage").Count+1),Position=p,Radius=.07f,Color=GasColor,Label="閉塞疑い / 同高度4方向に占有反応"});
                if(Alerts.FindAll(a=>a.Kind=="blockage").Count>=3)break;
            }
            Alerts.Sort((a,b)=>{int rank(string kind)=>kind=="person"?0:kind=="gas"?1:2;int result=rank(a.Kind).CompareTo(rank(b.Kind));return result!=0?result:string.CompareOrdinal(a.Id,b.Id);});
            var supported=new HashSet<Vector3Int>();
            foreach(var c in (JArray)reply["cells"])
            {var k=Key(c);float sensor=(float?)c["sensor_occupancy"]??(float)c["occupancy"];if((int)c["observations"]>=2&&sensor>.6f){if(InSlice(Knowledge.Center(k)))supported.Add(k);if(!firstEvidence.ContainsKey(k)){firstEvidence[k]=step;firstEvidenceTime[k]=Time.realtimeSinceStartup;}}}
            var nearby=new HashSet<Vector3Int>();foreach(var k in supported)for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++)nearby.Add(k+new Vector3Int(x,y,z));
            solidEvidence.Clear();foreach(var c in Knowledge.Cells.Values)if(c.Tier==RescueEvidenceTier.Measured&&c.Solid)solidEvidence.Add(c.Key);
            if(step>pulseStep)
            {pulseStep=step;pulseAt=Time.realtimeSinceStartup;pulseOrigins.Clear();if(reply["nodes"] is JArray emitters)for(int i=(int)(step/25)%5;i<emitters.Count&&pulseOrigins.Count<4;i+=Mathf.Max(1,emitters.Count/4))pulseOrigins.Add(RescueEvaluation.Vector(emitters[i]["estimated_position"]));}
            BuildSurfaces(supported,cell);BuildContext();
            var display=new Dictionary<Vector3Int,bool>();foreach(var c in (JArray)reply["cells"])if((float)c["occupancy"]>.6f){var k=Key(c);bool measured=supported.Contains(k);if(InSlice(Knowledge.Center(k))&&(measured||(ShowCompletion&&step>=500&&nearby.Contains(k))))display[k]=measured;}
            bool fine=(float?)reply["surface_resolution_m"]>0;FineObservedPoints=0;
            vertices.Clear();colors.Clear();indices.Clear();uv.Clear();VisibleObservedCells=VisibleCompletionCells=0;
            foreach(var entry in display)
            {
                var k=entry.Key;bool measured=entry.Value;if(measured)VisibleObservedCells++;else VisibleCompletionCells++;
                var center=(new Vector3(k.x,k.y,k.z)+Vector3.one*.5f)*cell;
                var color=measured?ObservedColor(center):new Color(.53f,.43f,.72f,.22f);
                if(measured)
                {
                    color.a*=Mathf.Clamp01((step-firstEvidence[k]+25)/125f);
                    if(fine)color.a*=.28f;
                }
                if(RenderSurfaces)color.a*=measured?.04f:0;
                for(int axis=0;axis<3;axis++)for(int sign=-1;sign<=1;sign+=2)
                {
                    var neighbor=k;neighbor[axis]+=sign;if(display.ContainsKey(neighbor))continue;int density=measured?2:1;
                    for(int u=0;u<density;u++)for(int v=0;v<density;v++)
                    {
                        var p=center;p[axis]+=sign*cell*.49f;p[(axis+1)%3]+=(u-(density-1)*.5f)*cell*.54f;p[(axis+2)%3]+=(v-(density-1)*.5f)*cell*.54f;
                        int hash=(k.x*73856093)^(k.y*19349663)^(k.z*83492791)^(axis*37+u*11+v*19);p[(axis+1)%3]+=((hash&1023)/1023f-.5f)*cell*.04f;Point(p,color,measured?(fine?.85f:1.9f):1.05f);
                    }
                }
            }
            if(fine&&reply["surface_points"] is JArray samples)foreach(JArray sample in samples)
            {
                var p=new Vector3((float)sample[0],(float)sample[1],(float)sample[2]);if(!InSlice(p))continue;var color=ObservedColor(p);
                color.a*=(float)sample[3]*Mathf.Clamp01((step-(float)sample[4]+25)/100f)*Mathf.Clamp01(1-(step-(float)sample[5])/1500f);
                if(RenderSurfaces)color.a=0;
                Point(p,color,2.15f);FineObservedPoints++;
            }
            SurfacePoints=vertices.Count/4;
            foreach(var node in (JArray)reply["nodes"])
            {
                string id=(string)node["node_id"];var p=RescueEvaluation.Vector(node["estimated_position"]);
                if(!trajectories.TryGetValue(id,out var history)){history=new List<Vector3>();trajectories[id]=history;}
                if(history.Count==0||Vector3.Distance(history[history.Count-1],p)>.012f){history.Add(p);if(history.Count>384)history.RemoveAt(0);}
                if(InSlice(p))Point(p,id==selected?Color.white:new Color(.78f,.87f,.90f,.6f),id==selected?5:3);
            }
            foreach(var alert in Alerts){if((alert.Kind=="person"&&!ShowPeople)||!InSlice(alert.Position))continue;Point(alert.Position,alert.Color,7);if(alert.Kind=="person"){Ring(alert,0);}else if(alert.Kind=="gas"||alert.Kind=="blockage"){Ring(alert,0);Triangle(alert);}else{Ring(alert,0);WaterWaves(alert);}}
            points.Clear();points.SetVertices(vertices);points.SetColors(colors);points.SetUVs(0,uv);points.SetTriangles(indices,0);points.RecalculateBounds();BuildTrail();AnimateEcho();
        }
        void Ring(AlertMark alert,int plane)
        {var p=new Vector3[65];for(int i=0;i<p.Length;i++){float a=i*Mathf.PI*2/64;p[i]=alert.Position+(plane==0?new Vector3(Mathf.Cos(a),0,Mathf.Sin(a)):new Vector3(Mathf.Cos(a),Mathf.Sin(a),0))*alert.Radius;}Stroke("Unconfirmed_"+alert.Kind,p,alert.Color,.0035f);}
        void Triangle(AlertMark a)=>Stroke("GasWarningTriangle",new[]{a.Position+new Vector3(0,.07f,0),a.Position+new Vector3(-.04f,0,0),a.Position+new Vector3(.04f,0,0),a.Position+new Vector3(0,.07f,0)},a.Color,.005f);
        void WaterWaves(AlertMark a)
        {for(int row=-1;row<=1;row++){var p=new Vector3[24];for(int i=0;i<p.Length;i++){float x=(i/23f-.5f)*a.Radius*1.4f;p[i]=a.Position+new Vector3(x,.005f,row*a.Radius*.33f+Mathf.Sin(i*.65f)*.008f);}Stroke("WaterWave",p,a.Color,.004f);}}
        void Stroke(string name,Vector3[] positions,Color color,float width)
        {var g=new GameObject(name);g.layer=8;g.transform.SetParent(Root.transform,false);var line=g.AddComponent<LineRenderer>();line.sharedMaterial=lineMaterial;line.startColor=line.endColor=color;line.startWidth=line.endWidth=width;line.positionCount=positions.Length;line.SetPositions(positions);markers.Add(g);}
        void BuildTrail()
        {
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();if(trajectories.TryGetValue(selected,out var positions))for(int i=1;i<positions.Count;i++)if(InSlice(positions[i])&&InSlice(positions[i-1]))Line(v,c,ix,positions[i-1],positions[i],new Color(.72f,.83f,.83f,.16f+i/(float)positions.Count*.2f));
            trail.Clear();trail.SetVertices(v);trail.SetColors(c);trail.SetIndices(ix,MeshTopology.Lines,0);trail.RecalculateBounds();
        }
        public void Dispose(){Knowledge.Dispose();if(Root)Object.DestroyImmediate(Root);Object.DestroyImmediate(sectionGuide);Object.DestroyImmediate(context);Object.DestroyImmediate(inferred);Object.DestroyImmediate(echo);Object.DestroyImmediate(points);Object.DestroyImmediate(surfaces);Object.DestroyImmediate(edges);Object.DestroyImmediate(surfaceMaterial);Object.DestroyImmediate(grid);Object.DestroyImmediate(trail);Object.DestroyImmediate(pointMaterial);Object.DestroyImmediate(lineMaterial);}
    }
}
