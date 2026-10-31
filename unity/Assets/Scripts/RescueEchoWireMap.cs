using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace Purumogu.Rescue
{
    /// A quiet green surface net. No points, markers, templates, fills, or animated effects.
    public sealed class RescueEchoWireMap : System.IDisposable
    {
        public readonly GameObject Root=new GameObject("ObservedGreenEchoMap");
        readonly Mesh surface=new Mesh{name="ObservedSurfaceDepth",indexFormat=IndexFormat.UInt32};
        readonly Mesh wire=new Mesh{name="ContinuousSurfaceQuadGrid",indexFormat=IndexFormat.UInt32};
        readonly Material depthMaterial,wireMaterial;
        readonly Dictionary<Vector3Int,float> settled=new Dictionary<Vector3Int,float>();
        public int InferredCells{get;private set;}
        public int PersonSegments{get;private set;}
        public int WaterSegments{get;private set;}
        public int GasSegments{get;private set;}
        public readonly struct WireZone
        {
            public readonly Vector3 Position;public readonly float Radius;public readonly int Kind;
            public WireZone(Vector3 position,float radius,int kind){Position=position;Radius=radius;Kind=kind;}
        }
        readonly List<WireZone> zones=new List<WireZone>();
        public int Quads{get;private set;}
        public int Segments{get;private set;}
        public Vector3[] ReconstructedVertices=>surface.vertices;
        public float MaximumRelaxation{get;private set;}
        public const float RelaxationLimitCells=.35f;
        static readonly Vector3Int[] Axes={Vector3Int.right,Vector3Int.up,new Vector3Int(0,0,1)};
        static readonly Vector3Int[] Corners={new Vector3Int(0,0,0),new Vector3Int(1,0,0),new Vector3Int(0,1,0),new Vector3Int(1,1,0),new Vector3Int(0,0,1),new Vector3Int(1,0,1),new Vector3Int(0,1,1),new Vector3Int(1,1,1)};
        static readonly int[,] CubeEdges={{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
        public RescueEchoWireMap()
        {
            depthMaterial=new Material(Resources.Load<Shader>("RescueEchoDepth"));
            wireMaterial=new Material(Resources.Load<Shader>("RescueEchoWire"));
            Add("DepthOnly",surface,depthMaterial);Add("GreenQuadGrid",wire,wireMaterial);
        }
        void Add(string name,Mesh mesh,Material material)
        {
            var child=new GameObject(name);child.layer=8;child.transform.SetParent(Root.transform,false);
            child.AddComponent<MeshFilter>().sharedMesh=mesh;child.AddComponent<MeshRenderer>().sharedMaterial=material;
        }
        public void SetViewport(float width,float height,float zoom=1)
        {wireMaterial.SetVector("_ViewportSize",new Vector4(Mathf.Max(1,width),Mathf.Max(1,height),0,0));}
        public void Update(JObject reply)
        {
            float cell=(float)reply["cell_m"];
            foreach(var item in (JArray)reply["cells"])
            {
                int count=(int)item["observations"];if(count<2||item["sensor_occupancy"]==null||item["sensor_occupancy"].Type==JTokenType.Null)continue;
                float value=(float)item["sensor_occupancy"];var k=item["key"];var key=new Vector3Int((int)k[0],(int)k[1],(int)k[2]);
                // Display hysteresis avoids a wire surface trembling with small probability changes.
                if(settled.TryGetValue(key,out float previous))
                {if(count>=4&&((previous>.5f&&value<=.20f)||(previous<.5f&&value>=.80f)))settled[key]=value>.5f?.95f:.05f;}
                else if(value>=.70f||value<=.30f)settled[key]=value>.5f?.95f:.05f;
            }
            zones.Clear();
            if(reply["cues"] is JArray cues)foreach(var cue in cues)zones.Add(new WireZone(RescueEvaluation.Vector(cue["position"]),Mathf.Clamp((float)cue["radius_m"],.15f,.32f),1));
            if(reply["hazards"] is JArray hazards)foreach(var hazard in hazards)
                zones.Add(new WireZone(RescueEvaluation.Vector(hazard["position"]),Mathf.Clamp((float)hazard["radius_m"],.09f,.4f),(string)hazard["kind"]=="gas"?3:2));
            var merged=new Dictionary<Vector3Int,float>(settled);var inferred=new HashSet<Vector3Int>();
            foreach(var item in (JArray)reply["cells"])
            {
                if((string)item["source"]!="local_surface_completion")continue;var k=item["key"];var key=new Vector3Int((int)k[0],(int)k[1],(int)k[2]);
                if(merged.ContainsKey(key))continue;merged[key]=(float)item["occupancy"];inferred.Add(key);
            }
            InferredCells=inferred.Count;PersonSegments=WaterSegments=GasSegments=0;
            var builder=new SurfaceNet(merged,cell,inferred);
            builder.Build();Quads=builder.Quads.Count;MaximumRelaxation=builder.MaximumRelaxation;
            var triangles=new List<int>();var segments=new HashSet<(int,int)>();
            foreach(var face in builder.Quads)
            {
                triangles.AddRange(new[]{face[0],face[1],face[2],face[0],face[2],face[3]});
                for(int i=0;i<4;i++){int a=face[i],b=face[(i+1)%4];segments.Add(a<b?(a,b):(b,a));}
            }
            surface.Clear();surface.SetVertices(builder.Positions);surface.SetTriangles(triangles,0);surface.RecalculateBounds();
            var vertices=new List<Vector3>();var tangents=new List<Vector3>();var sides=new List<Vector2>();var estimates=new List<Vector2>();var colors=new List<Color>();var indices=new List<int>();
            foreach(var segment in segments)
            {
                var a=builder.Positions[segment.Item1];var b=builder.Positions[segment.Item2];if((a-b).sqrMagnitude<1e-10f)continue;
                int index=vertices.Count;var tangent=b-a;
                vertices.AddRange(new[]{a,a,b,b});tangents.AddRange(new[]{tangent,tangent,tangent,tangent});
                bool estimated=builder.Estimated[segment.Item1]||builder.Estimated[segment.Item2];float length=tangent.magnitude;
                sides.AddRange(new[]{new Vector2(-1,0),new Vector2(1,0),new Vector2(-1,length),new Vector2(1,length)});
                var flag=new Vector2(estimated?1:0,0);estimates.AddRange(new[]{flag,flag,flag,flag});
                int kind=ZoneAt((a+b)*.5f);Color color=kind==3?new Color(.95f,.025f,.09f,.95f):kind==2?new Color(.025f,.37f,.95f,.95f):kind==1?new Color(1,.52f,.025f,.98f):new Color(.025f,.62f,.075f,.84f);
                if(estimated)color.a*=.48f;colors.AddRange(new[]{color,color,color,color});
                if(kind==1)PersonSegments++;else if(kind==2)WaterSegments++;else if(kind==3)GasSegments++;
                indices.AddRange(new[]{index,index+1,index+2,index+2,index+1,index+3});
            }
            Segments=vertices.Count/4;
            wire.Clear();wire.SetVertices(vertices);wire.SetNormals(tangents);wire.SetUVs(0,sides);wire.SetUVs(1,estimates);wire.SetColors(colors);wire.SetTriangles(indices,0);wire.RecalculateBounds();
        }
        int ZoneAt(Vector3 point)
        {
            bool person=false;int danger=0;
            foreach(var zone in zones)
            {
                if(Mathf.Abs(point.y-zone.Position.y)>.17f)continue;
                if(new Vector2(point.x-zone.Position.x,point.z-zone.Position.z).magnitude>zone.Radius)continue;
                if(zone.Kind==1)person=true;else danger=Mathf.Max(danger,zone.Kind);
            }
            if(person&&danger>0){int cellParity=Mathf.FloorToInt(point.x/.05f)+Mathf.FloorToInt(point.z/.05f);return (cellParity&1)==0?1:danger;}
            return danger>0?danger:person?1:0;
        }
        sealed class SurfaceNet
        {
            readonly Dictionary<Vector3Int,float> evidence;
            readonly Dictionary<Vector3Int,int> vertexIds=new Dictionary<Vector3Int,int>();
            readonly HashSet<Vector3Int> rejected=new HashSet<Vector3Int>();
            readonly float cell;
            readonly HashSet<Vector3Int> inferred;
            public readonly List<bool> Estimated=new List<bool>();
            public readonly List<Vector3> Positions=new List<Vector3>();
            public readonly List<int[]> Quads=new List<int[]>();
            public float MaximumRelaxation;
            public SurfaceNet(Dictionary<Vector3Int,float> evidence,float cell,HashSet<Vector3Int> inferred){this.evidence=evidence;this.cell=cell;this.inferred=inferred;}
            Vector3 Point(Vector3Int k)=>((Vector3)k+Vector3.one*.5f)*cell;
            int Vertex(Vector3Int cube)
            {
                if(vertexIds.TryGetValue(cube,out int existing))return existing;if(rejected.Contains(cube))return -1;
                var known=new bool[8];var values=new float[8];int count=0;
                for(int i=0;i<8;i++)if(evidence.TryGetValue(cube+Corners[i],out values[i])){known[i]=true;count++;}
                if(count<4){rejected.Add(cube);return -1;}
                var sum=Vector3.zero;int crossings=0;
                for(int edge=0;edge<12;edge++)
                {
                    int a=CubeEdges[edge,0],b=CubeEdges[edge,1];if(!known[a]||!known[b]||(values[a]>.5f)==(values[b]>.5f))continue;
                    float fraction=(.5f-values[a])/(values[b]-values[a]);sum+=Vector3.Lerp(Point(cube+Corners[a]),Point(cube+Corners[b]),fraction);crossings++;
                }
                if(crossings==0){rejected.Add(cube);return -1;}
                int id=Positions.Count;Positions.Add(sum/crossings);bool estimate=false;foreach(var corner in Corners)if(inferred.Contains(cube+corner))estimate=true;Estimated.Add(estimate);vertexIds[cube]=id;return id;
            }
            public void Build()
            {
                foreach(var sample in evidence)for(int axis=0;axis<3;axis++)
                {
                    var k=sample.Key;if(!evidence.TryGetValue(k+Axes[axis],out float other)||(sample.Value>.5f)==(other>.5f))continue;
                    var a=Axes[(axis+1)%3];var b=Axes[(axis+2)%3];
                    var face=new[]{Vertex(k-a-b),Vertex(k-b),Vertex(k),Vertex(k-a)};
                    if(System.Array.Exists(face,index=>index<0))continue;Quads.Add(face);
                }
                // Relax the connected surface, not independent voxel boxes. Keep holes at observation frontiers.
                var adjacent=new List<HashSet<int>>();for(int i=0;i<Positions.Count;i++)adjacent.Add(new HashSet<int>());
                foreach(var face in Quads)for(int i=0;i<4;i++){int a=face[i],b=face[(i+1)%4];adjacent[a].Add(b);adjacent[b].Add(a);}
                var original=Positions.ToArray();
                for(int iteration=0;iteration<2;iteration++)
                {
                    var next=Positions.ToArray();
                    for(int i=0;i<Positions.Count;i++)
                    {
                        if(adjacent[i].Count<3)continue;var average=Vector3.zero;
                        foreach(int neighbor in adjacent[i])average+=Positions[neighbor];average/=adjacent[i].Count;
                        var candidate=Vector3.Lerp(Positions[i],average,.42f);candidate=original[i]+Vector3.ClampMagnitude(candidate-original[i],cell*RelaxationLimitCells);
                        if(evidence.ContainsKey(Vector3Int.FloorToInt(candidate/cell)))next[i]=candidate;
                    }
                    for(int i=0;i<Positions.Count;i++)Positions[i]=next[i];
                }
                for(int i=0;i<Positions.Count;i++)MaximumRelaxation=Mathf.Max(MaximumRelaxation,Vector3.Distance(original[i],Positions[i]));
            }
        }
        public void Dispose()
        {if(Root)Object.DestroyImmediate(Root);Object.DestroyImmediate(surface);Object.DestroyImmediate(wire);Object.DestroyImmediate(depthMaterial);Object.DestroyImmediate(wireMaterial);}
    }
}
