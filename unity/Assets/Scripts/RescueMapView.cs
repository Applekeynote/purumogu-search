using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace Purumogu.Rescue
{
    /// Render only the server's reconstructed cells, poses and unconfirmed cues.
    public sealed class RescueMapView : System.IDisposable
    {
        public readonly GameObject Root=new GameObject("InferredMapOnly");
        public readonly List<Vector3> CuePositions=new List<Vector3>();
        public readonly List<string> CueLabels=new List<string>();
        readonly Mesh geometry=new Mesh{name="SensorOccupancyWireframe",indexFormat=IndexFormat.UInt32};
        readonly Material material;
        readonly List<GameObject> markers=new List<GameObject>();
        readonly Dictionary<Vector3Int,long> firstEvidenceStep=new Dictionary<Vector3Int,long>();
        public int VisibleObservedCells{get;private set;}
        public int VisibleCompletionCells{get;private set;}
        public RescueMapView()
        {
            material=new Material(Shader.Find("Sprites/Default"));
            var g=new GameObject("ObservedSurface");g.layer=8;g.transform.SetParent(Root.transform,false);g.AddComponent<MeshFilter>().sharedMesh=geometry;g.AddComponent<MeshRenderer>().sharedMaterial=material;
            var vertices=new List<Vector3>();var colors=new List<Color>();var indices=new List<int>();
            for(int i=-15;i<=15;i+=3)Line(vertices,colors,indices,new Vector3(i*.05f,-.025f,-.3f),new Vector3(i*.05f,-.025f,3.2f),new Color(.1f,.28f,.32f,.6f));
            for(int i=-6;i<=64;i+=3)Line(vertices,colors,indices,new Vector3(-.75f,-.025f,i*.05f),new Vector3(.75f,-.025f,i*.05f),new Color(.1f,.28f,.32f,.6f));
            var grid=new GameObject("MapCoordinateGrid");grid.layer=8;grid.transform.SetParent(Root.transform,false);var mesh=new Mesh();mesh.SetVertices(vertices);mesh.SetColors(colors);mesh.SetIndices(indices,MeshTopology.Lines,0);grid.AddComponent<MeshFilter>().sharedMesh=mesh;grid.AddComponent<MeshRenderer>().sharedMaterial=material;
        }
        static void Line(List<Vector3> vertices,List<Color> colors,List<int> indices,Vector3 a,Vector3 b,Color color)
        {
            int n=vertices.Count;vertices.Add(a);vertices.Add(b);colors.Add(color);colors.Add(color);indices.Add(n);indices.Add(n+1);
        }
        public void Update(JObject reply)
        {
            var vertices=new List<Vector3>();var colors=new List<Color>();var indices=new List<int>();float cell=(float)reply["cell_m"];
            long step=(long)reply["sim_step"];
            // Reveal from repeated ray evidence. A globally selected shape family must
            // not appear as if its unobserved walls had already been scanned.
            var supported=new HashSet<Vector3Int>();
            foreach(var c in (JArray)reply["cells"])
            {
                var k=(JArray)c["key"];var gridKey=new Vector3Int((int)k[0],(int)k[1],(int)k[2]);
                float sensor=(float?)c["sensor_occupancy"]??(float)c["occupancy"];
                if((int)c["observations"]>=2&&sensor>.6f){supported.Add(gridKey);if(!firstEvidenceStep.ContainsKey(gridKey))firstEvidenceStep[gridKey]=step;}
            }
            var nearby=new HashSet<Vector3Int>();
            foreach(var k in supported)for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int dz=-1;dz<=1;dz++)nearby.Add(k+new Vector3Int(dx,dy,dz));
            var display=new Dictionary<Vector3Int,bool>();VisibleObservedCells=VisibleCompletionCells=0;
            var occupied=new HashSet<Vector3Int>();
            foreach(var c in (JArray)reply["cells"])if((float)c["occupancy"]>.6f)
            {
                var k=(JArray)c["key"];var gridKey=new Vector3Int((int)k[0],(int)k[1],(int)k[2]);bool measured=supported.Contains(gridKey);
                if(!measured&&!(step>=500&&nearby.Contains(gridKey)))continue;
                occupied.Add(gridKey);display[gridKey]=measured;if(measured)VisibleObservedCells++;else VisibleCompletionCells++;
            }
            int free=0;
            foreach(var token in (JArray)reply["cells"])
            {
                float p=(float)token["occupancy"];var key=(JArray)token["key"];var center=new Vector3(((int)key[0]+.5f)*cell,((int)key[1]+.5f)*cell,((int)key[2]+.5f)*cell);
                var gridKey=new Vector3Int((int)key[0],(int)key[1],(int)key[2]);
                if(p>.60f)
                {
                    if(!display.TryGetValue(gridKey,out bool measured))continue;
                    float half=cell*.47f;Color color=Color.Lerp(new Color(.05f,.65f,.84f,.35f),new Color(.3f,1,.82f,1),Mathf.Clamp01((p-.6f)*2.5f));color=Color.Lerp(color,new Color(.65f,.5f,1,.8f),Mathf.Clamp01(center.y/.8f)*.6f);
                    if(!measured)color=new Color(.62f,.45f,1,.18f);
                    else color.a*=Mathf.Clamp01((step-firstEvidenceStep[gridKey]+25)/150f);
                    for(int axis=0;axis<3;axis++)for(int sign=-1;sign<=1;sign+=2)
                    {
                        var neighbor=gridKey;neighbor[axis]+=sign;if(occupied.Contains(neighbor))continue;
                        for(int edge=0;edge<4;edge++)
                        {
                            int along=(axis+1+edge%2)%3,fixedAxis=(axis+2-edge%2+3)%3;var start=center;var end=center;start[axis]+=sign*half;end[axis]+=sign*half;start[along]-=half;end[along]+=half;float side=edge<2?-1:1;start[fixedAxis]+=side*half;end[fixedAxis]+=side*half;Line(vertices,colors,indices,start,end,color);
                        }
                    }
                }
                else if(p<.4f&&(int)token["observations"]>=2&&free++%12==0)
                {
                    Line(vertices,colors,indices,center-Vector3.right*cell*.12f,center+Vector3.right*cell*.12f,new Color(.15f,.55f,.55f,.2f));
                }
            }
            geometry.Clear();geometry.SetVertices(vertices);geometry.SetColors(colors);geometry.SetIndices(indices,MeshTopology.Lines,0);geometry.RecalculateBounds();
            foreach(var marker in markers)if(marker)Object.DestroyImmediate(marker);markers.Clear();CuePositions.Clear();CueLabels.Clear();
            foreach(var node in (JArray)reply["nodes"])
            {
                var sphere=GameObject.CreatePrimitive(PrimitiveType.Sphere);sphere.name="Estimated_"+(string)node["node_id"];sphere.layer=8;sphere.transform.SetParent(Root.transform,false);sphere.transform.position=RescueEvaluation.Vector(node["estimated_position"]);sphere.transform.localScale=Vector3.one*.018f;Object.DestroyImmediate(sphere.GetComponent<Collider>());sphere.GetComponent<Renderer>().sharedMaterial=material;markers.Add(sphere);
            }
            int number=0;
            foreach(var cue in (JArray)reply["cues"])
            {
                Vector3 center=RescueEvaluation.Vector(cue["position"]);float radius=(float)cue["radius_m"];
                CuePositions.Add(center);CueLabels.Add($"候補 {(char)('A'+number++)} / 未確認\n優先度 {(float)cue["score"]:F2} · {(int)cue["observers"]} ノード\n探索半径 約{radius*100:F0} cm");
                for(int plane=0;plane<3;plane++)
                {
                    var ring=new GameObject("UnconfirmedAcousticEnvelope");ring.layer=8;ring.transform.SetParent(Root.transform,false);var line=ring.AddComponent<LineRenderer>();line.sharedMaterial=material;line.startColor=line.endColor=new Color(1,.63f,.22f,.8f);line.startWidth=line.endWidth=.003f;line.positionCount=65;
                    for(int i=0;i<=64;i++){float a=i*Mathf.PI*2/64;Vector3 p=plane==0?new Vector3(Mathf.Cos(a),0,Mathf.Sin(a)):plane==1?new Vector3(Mathf.Cos(a),Mathf.Sin(a),0):new Vector3(0,Mathf.Cos(a),Mathf.Sin(a));line.SetPosition(i,center+p*radius);}
                    markers.Add(ring);
                }
            }
        }
        public void Dispose(){if(Root)Object.DestroyImmediate(Root);Object.DestroyImmediate(geometry);Object.DestroyImmediate(material);}
    }
}
