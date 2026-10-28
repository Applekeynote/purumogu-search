using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace Purumogu.Rescue
{
    /// Fine observed endpoints overlay a faint 5cm evidence scaffold; neither is ground truth.
    public sealed class RescueMapView : System.IDisposable
    {
        public static readonly Color PersonColor=new Color(1,.70f,.18f),GasColor=new Color(1,.30f,.48f),WaterColor=new Color(.32f,.64f,1);
        public sealed class AlertMark {public string Kind,Id,Label;public Vector3 Position;public float Radius;public Color Color;}
        public readonly List<AlertMark> Alerts=new List<AlertMark>();
        public readonly List<Vector3> CuePositions=new List<Vector3>();
        public readonly List<string> CueLabels=new List<string>();
        public readonly GameObject Root=new GameObject("InferredMapOnly");
        readonly Mesh points=new Mesh{name="ObservedVoxelSurfacePoints",indexFormat=IndexFormat.UInt32};
        readonly Mesh grid=new Mesh{name="QuietCoordinateGrid"},trail=new Mesh{name="EstimatedSelectedNodeTrail"};
        readonly Material pointMaterial,lineMaterial;
        readonly List<Vector3> vertices=new List<Vector3>(),uv=new List<Vector3>();
        readonly List<Color> colors=new List<Color>();
        readonly List<int> indices=new List<int>();
        readonly List<GameObject> markers=new List<GameObject>();
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
            MeshObject("ObservedSurfacePoints",points,pointMaterial);MeshObject("MapGrid",grid,lineMaterial);MeshObject("NodeEstimatedTrail",trail,lineMaterial);points.MarkDynamic();trail.MarkDynamic();
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();var faint=new Color(.19f,.25f,.28f,.26f);
            for(int x=-8;x<=8;x+=2)Line(v,c,ix,new Vector3(x*.1f,-.035f,-.3f),new Vector3(x*.1f,-.035f,3.2f),faint);
            for(int z=-3;z<=32;z+=2)Line(v,c,ix,new Vector3(-.8f,-.035f,z*.1f),new Vector3(.8f,-.035f,z*.1f),faint);
            grid.SetVertices(v);grid.SetColors(c);grid.SetIndices(ix,MeshTopology.Lines,0);grid.RecalculateBounds();
        }
        void MeshObject(string name,Mesh mesh,Material material)
        {var g=new GameObject(name);g.layer=8;g.transform.SetParent(Root.transform,false);g.AddComponent<MeshFilter>().sharedMesh=mesh;g.AddComponent<MeshRenderer>().sharedMaterial=material;}
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
            var color=Color.Lerp(new Color(.17f,.43f,.45f,.86f),new Color(.57f,.78f,.73f,.95f),Mathf.Clamp01(center.y/.7f));
            foreach(var alert in Alerts)if(alert.Kind!="person"&&Mathf.Abs(center.y-alert.Position.y)<.17f&&new Vector2(center.x-alert.Position.x,center.z-alert.Position.z).magnitude<alert.Radius){color=alert.Color;color.a=.95f;if(alert.Kind=="gas")break;}
            return color;
        }
        public void Update(JObject reply)
        {
            float cell=(float)reply["cell_m"];long step=(long)reply["sim_step"];
            foreach(var marker in markers)if(marker)Object.DestroyImmediate(marker);markers.Clear();Alerts.Clear();CueLabels.Clear();CuePositions.Clear();
            GasZones=WaterZones=0;GasObservedNodes=(int?)reply["gas_observed_nodes"]??0;WaterObservedNodes=(int?)reply["water_observed_nodes"]??0;
            int number=0;
            foreach(var cue in (JArray)reply["cues"])
            {
                var p=RescueEvaluation.Vector(cue["position"]);float radius=(float)cue["radius_m"];string id=((char)('A'+number++)).ToString();
                var label=$"人候補 {id} / 未確認  ·  {(int)cue["observers"]}ノード  ·  優先度 {(float)cue["score"]:F2}";
                CuePositions.Add(p);CueLabels.Add(label);Alerts.Add(new AlertMark{Kind="person",Id=id,Label=label,Position=p,Radius=radius,Color=PersonColor});
            }
            if(reply["hazards"] is JArray hazards)foreach(var hazard in hazards)
            {
                bool gas=(string)hazard["kind"]=="gas";string id=gas?"G"+(++GasZones):"W"+(++WaterZones);float value=(float)hazard["value"];int observers=(int)hazard["observers"];
                Alerts.Add(new AlertMark{Kind=gas?"gas":"water",Id=id,Position=RescueEvaluation.Vector(hazard["position"]),Radius=(float)hazard["radius_m"],Color=gas?GasColor:WaterColor,
                    Label=gas?$"ガス反応 {id}  ·  相対値 {value:F2}  ·  {observers}ノード":$"浸水 {id}  ·  水深 {value*100:F1}cm  ·  {observers}ノード"});
            }
            Alerts.Sort((a,b)=>{int rank(string kind)=>kind=="person"?0:kind=="gas"?1:2;int result=rank(a.Kind).CompareTo(rank(b.Kind));return result!=0?result:string.CompareOrdinal(a.Id,b.Id);});
            var supported=new HashSet<Vector3Int>();
            foreach(var c in (JArray)reply["cells"])
            {var k=Key(c);float sensor=(float?)c["sensor_occupancy"]??(float)c["occupancy"];if((int)c["observations"]>=2&&sensor>.6f){supported.Add(k);if(!firstEvidence.ContainsKey(k))firstEvidence[k]=step;}}
            var nearby=new HashSet<Vector3Int>();foreach(var k in supported)for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++)nearby.Add(k+new Vector3Int(x,y,z));
            var display=new Dictionary<Vector3Int,bool>();foreach(var c in (JArray)reply["cells"])if((float)c["occupancy"]>.6f){var k=Key(c);bool measured=supported.Contains(k);if(measured||(step>=500&&nearby.Contains(k)))display[k]=measured;}
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
                var p=new Vector3((float)sample[0],(float)sample[1],(float)sample[2]);var color=ObservedColor(p);
                color.a*=(float)sample[3]*Mathf.Clamp01((step-(float)sample[4]+25)/100f)*Mathf.Clamp01(1-(step-(float)sample[5])/1500f);
                Point(p,color,2.15f);FineObservedPoints++;
            }
            SurfacePoints=vertices.Count/4;
            foreach(var node in (JArray)reply["nodes"])
            {
                string id=(string)node["node_id"];var p=RescueEvaluation.Vector(node["estimated_position"]);
                if(!trajectories.TryGetValue(id,out var history)){history=new List<Vector3>();trajectories[id]=history;}
                if(history.Count==0||Vector3.Distance(history[history.Count-1],p)>.012f){history.Add(p);if(history.Count>384)history.RemoveAt(0);}
                Point(p,id==selected?Color.white:new Color(.78f,.87f,.90f,.6f),id==selected?5:3);
            }
            foreach(var alert in Alerts){Point(alert.Position,alert.Color,7);if(alert.Kind=="person"){Ring(alert,0);Ring(alert,1);}else if(alert.Kind=="gas"){Ring(alert,0);Triangle(alert);}else{Ring(alert,0);WaterWaves(alert);}}
            points.Clear();points.SetVertices(vertices);points.SetColors(colors);points.SetUVs(0,uv);points.SetTriangles(indices,0);points.RecalculateBounds();BuildTrail();
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
            var v=new List<Vector3>();var c=new List<Color>();var ix=new List<int>();if(trajectories.TryGetValue(selected,out var positions))for(int i=1;i<positions.Count;i++)Line(v,c,ix,positions[i-1],positions[i],new Color(.72f,.83f,.83f,.16f+i/(float)positions.Count*.2f));
            trail.Clear();trail.SetVertices(v);trail.SetColors(c);trail.SetIndices(ix,MeshTopology.Lines,0);trail.RecalculateBounds();
        }
        public void Dispose(){if(Root)Object.DestroyImmediate(Root);Object.DestroyImmediate(points);Object.DestroyImmediate(grid);Object.DestroyImmediate(trail);Object.DestroyImmediate(pointMaterial);Object.DestroyImmediate(lineMaterial);}
    }
}
