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
            var builder=new SurfaceNet(settled,cell);
            builder.Build();Quads=builder.Quads.Count;MaximumRelaxation=builder.MaximumRelaxation;
            var triangles=new List<int>();var segments=new HashSet<(int,int)>();
            foreach(var face in builder.Quads)
            {
                triangles.AddRange(new[]{face[0],face[1],face[2],face[0],face[2],face[3]});
                for(int i=0;i<4;i++){int a=face[i],b=face[(i+1)%4];segments.Add(a<b?(a,b):(b,a));}
            }
            surface.Clear();surface.SetVertices(builder.Positions);surface.SetTriangles(triangles,0);surface.RecalculateBounds();
            var vertices=new List<Vector3>();var tangents=new List<Vector3>();var sides=new List<Vector2>();var indices=new List<int>();
            foreach(var segment in segments)
            {
                var a=builder.Positions[segment.Item1];var b=builder.Positions[segment.Item2];if((a-b).sqrMagnitude<1e-10f)continue;
                int index=vertices.Count;var tangent=b-a;
                vertices.AddRange(new[]{a,a,b,b});tangents.AddRange(new[]{tangent,tangent,tangent,tangent});
                sides.AddRange(new[]{new Vector2(-1,0),new Vector2(1,0),new Vector2(-1,0),new Vector2(1,0)});
                indices.AddRange(new[]{index,index+1,index+2,index+2,index+1,index+3});
            }
            Segments=vertices.Count/4;
            wire.Clear();wire.SetVertices(vertices);wire.SetNormals(tangents);wire.SetUVs(0,sides);wire.SetTriangles(indices,0);wire.RecalculateBounds();
        }
        sealed class SurfaceNet
        {
            readonly Dictionary<Vector3Int,float> evidence;
            readonly Dictionary<Vector3Int,int> vertexIds=new Dictionary<Vector3Int,int>();
            readonly HashSet<Vector3Int> rejected=new HashSet<Vector3Int>();
            readonly float cell;
            public readonly List<Vector3> Positions=new List<Vector3>();
            public readonly List<int[]> Quads=new List<int[]>();
            public float MaximumRelaxation;
            public SurfaceNet(Dictionary<Vector3Int,float> evidence,float cell){this.evidence=evidence;this.cell=cell;}
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
                int id=Positions.Count;Positions.Add(sum/crossings);vertexIds[cube]=id;return id;
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
