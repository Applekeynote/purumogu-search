using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueScene
    {
        static float Cross(Vector2 a,Vector2 b)=>a.x*b.y-a.y*b.x;
        static float PolygonArea(IReadOnlyList<Vector2> p)
        {float sum=0;for(int i=0;i<p.Count;i++)sum+=Cross(p[i],p[(i+1)%p.Count]);return Mathf.Abs(sum)*.5f;}
        static Vector2[] RectanglePolygon(Rect r)=>new[]{new Vector2(r.xMin,r.yMin),new Vector2(r.xMax,r.yMin),new Vector2(r.xMax,r.yMax),new Vector2(r.xMin,r.yMax)};
        static List<Vector2> ClipFloor(IReadOnlyList<Vector2> polygon,Vector2 a,Vector2 b,bool inside)
        {
            var output=new List<Vector2>();if(polygon.Count==0)return output;
            Vector2 previous=polygon[polygon.Count-1];float before=Cross(b-a,previous-a);
            foreach(var current in polygon)
            {
                float after=Cross(b-a,current-a);bool keep=inside?after>=0:after<=0,kept=inside?before>=0:before<=0;
                if(keep!=kept)
                {float t=before/(before-after);output.Add(Vector2.LerpUnclamped(previous,current,t));}
                if(keep)output.Add(current);previous=current;before=after;
            }
            for(int i=output.Count-1;i>=0&&output.Count>1;i--)if((output[i]-output[(i+1)%output.Count]).sqrMagnitude<1e-12f)output.RemoveAt(i);
            return output;
        }
        static List<Vector2[]> SubtractOpening(Vector2[] polygon,Vector2[] opening)
        {
            var result=new List<Vector2[]>();var remaining=polygon.ToList();
            for(int i=0;i<opening.Length&&remaining.Count>=3;i++)
            {
                var a=opening[i];var b=opening[(i+1)%opening.Length];
                var outside=ClipFloor(remaining,a,b,false);
                if(outside.Count>=3&&PolygonArea(outside)>1e-7f)result.Add(outside.ToArray());
                remaining=ClipFloor(remaining,a,b,true);
            }
            return result;
        }
        void OpenRampLanding(int upperLevel,Vector3 start,Vector3 end)
        {
            Vector2 a=new Vector2(start.x,start.z),b=new Vector2(end.x,end.z);
            var direction=(b-a).normalized;var side=new Vector2(-direction.y,direction.x)*.16f;
            // Keep the real landing beyond the ramp end. Only cut the headroom
            // above its upper part; a box through an intact floor is not a stairway.
            float fraction=Mathf.Clamp01(1-(.036f+.07f)/Mathf.Max(.01f,end.y-start.y));
            var entry=Vector2.Lerp(a,b,fraction);
            var opening=new[]{entry-side,b-side,b+side,entry+side};
            foreach(var zone in FloorZones.Where(f=>Mathf.Abs(f.Height-upperLevel*.28f)<.001f).ToArray())
            {
                var polygon=zone.Polygon??RectanglePolygon(zone.Bounds);var pieces=SubtractOpening(polygon,opening);
                float remaining=pieces.Sum(p=>PolygonArea(p));
                if(PolygonArea(polygon)-remaining<1e-7f)continue;
                if(zone.Surface)
                {
                    var renderer=zone.Surface.GetComponent<Renderer>();Roofs.Remove(renderer);
                    roofEdges.RemoveAll(edge=>edge&&edge.transform.IsChildOf(zone.Surface.transform));
                    UnityEngine.Object.DestroyImmediate(zone.Surface);
                }
                FloorZones.Remove(zone);
                foreach(var piece in pieces)CreateFloorPiece(piece,zone.Height,upperLevel);
            }
        }
        void CreateFloorPiece(Vector2[] polygon,float height,int level)
        {
            int n=polygon.Length;var vertices=new List<Vector3>();var triangles=new List<int>();
            foreach(var p in polygon)vertices.Add(new Vector3(p.x,height,p.y));
            foreach(var p in polygon)vertices.Add(new Vector3(p.x,height-.036f,p.y));
            for(int i=1;i<n-1;i++)triangles.AddRange(new[]{0,i+1,i,n,n+i,n+i+1});
            for(int i=0;i<n;i++){int j=(i+1)%n;triangles.AddRange(new[]{i,j,n+i,j,n+j,n+i});}
            var mesh=new Mesh{name="FloorWithRampOpening"};mesh.SetVertices(vertices);mesh.SetTriangles(triangles,0);mesh.RecalculateNormals();mesh.RecalculateBounds();resources.Add(mesh);
            var surface=new GameObject("ShapedFloor_L"+level);surface.layer=6;surface.transform.SetParent(Root.transform,false);
            surface.AddComponent<MeshFilter>().sharedMesh=mesh;var renderer=surface.AddComponent<MeshRenderer>();renderer.sharedMaterial=Concrete;Roofs.Add(renderer);
            var collider=surface.AddComponent<MeshCollider>();collider.sharedMesh=mesh;collider.convex=true;collider.sharedMaterial=Contact;
            for(int i=0;i<n;i++)
            {
                int j=(i+1)%n;
                foreach(var pair in new[]{new[]{i,j},new[]{n+i,n+j},new[]{i,n+i}})
                {
                    var g=new GameObject("SlabContactOutline");g.transform.SetParent(surface.transform,false);var line=g.AddComponent<LineRenderer>();line.useWorldSpace=true;line.positionCount=2;line.widthMultiplier=.006f;line.sharedMaterial=roofEdgeMaterial;line.SetPosition(0,vertices[pair[0]]);line.SetPosition(1,vertices[pair[1]]);roofEdges.Add(line);
                }
            }
            float left=polygon.Min(p=>p.x),right=polygon.Max(p=>p.x),bottom=polygon.Min(p=>p.y),top=polygon.Max(p=>p.y);
            FloorZones.Add(new FloorZone{Bounds=Rect.MinMaxRect(left,bottom,right,top),Height=height,Polygon=polygon,Surface=surface});
        }
    }
}
