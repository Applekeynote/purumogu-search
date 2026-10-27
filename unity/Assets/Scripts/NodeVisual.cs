using System.Collections.Generic;
using UnityEngine;

namespace Purumogu {
    /// Visual skin only. The original core BoxCollider and capture-arm colliders are unchanged.
    public static class NodeVisual {
        static Mesh plate;
        public static void Attach(Transform root,Material shell,Material accent){
            if(!plate)plate=RoundedPlate();
            var face=new GameObject("IvoryPlate");face.transform.SetParent(root,false);face.AddComponent<MeshFilter>().sharedMesh=plate;face.AddComponent<MeshRenderer>().sharedMaterial=shell;
            var indicator=GameObject.CreatePrimitive(PrimitiveType.Cube);indicator.name="StatusStripVisual";indicator.transform.SetParent(root,false);indicator.transform.localPosition=new Vector3(0,.0102f,.002f);indicator.transform.localScale=new Vector3(.014f,.001f,.003f);Object.DestroyImmediate(indicator.GetComponent<Collider>());indicator.GetComponent<Renderer>().sharedMaterial=accent;
        }
        static Mesh RoundedPlate(){
            const float hx=.018f,hz=.021f,r=.0045f,halfY=.0095f;var outline=new List<Vector2>();
            for(int corner=0;corner<4;corner++){float cx=corner<2?hx-r:-hx+r,cz=(corner==0||corner==3)?hz-r:-hz+r;float start=90-corner*90;
                for(int j=0;j<=6;j++){float a=(start-j*15)*Mathf.Deg2Rad;outline.Add(new Vector2(cx+Mathf.Cos(a)*r,cz+Mathf.Sin(a)*r));}}
            var vertices=new List<Vector3>();var triangles=new List<int>();int n=outline.Count;
            vertices.Add(new Vector3(0,halfY,0));vertices.Add(new Vector3(0,-halfY,0));
            for(int i=0;i<n;i++){var p=outline[i];vertices.Add(new Vector3(p.x,halfY,p.y));vertices.Add(new Vector3(p.x,-halfY,p.y));}
            for(int i=0;i<n;i++){int a=2+i*2,b=2+((i+1)%n)*2;triangles.AddRange(new[]{0,a,b,1,b+1,a+1});}
            // Separate side vertices preserve a flat top while rounding the silhouette.
            for(int i=0;i<n;i++){var a=outline[i];var b=outline[(i+1)%n];int v=vertices.Count;vertices.AddRange(new[]{new Vector3(a.x,-halfY,a.y),new Vector3(a.x,halfY,a.y),new Vector3(b.x,halfY,b.y),new Vector3(b.x,-halfY,b.y)});triangles.AddRange(new[]{v,v+2,v+1,v,v+3,v+2});}
            var m=new Mesh{name="RoundedMicrobotPlate"};m.SetVertices(vertices);m.SetTriangles(triangles,0);m.RecalculateNormals();m.RecalculateBounds();return m;
        }
    }
}
