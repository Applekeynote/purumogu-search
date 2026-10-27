using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu {
    /// Geometry and labels live here; neither is serialized by the observation adapter.
    public sealed class ScenarioBuilder {
        public static readonly string[] Families={"STRAIGHT","T_BRANCH","DEAD_END","REJOIN","NARROWING","OTHER"};
        public GameObject Root {get;private set;}
        public float Cell {get;private set;}
        public float Friction {get;private set;}
        public int Family {get;private set;}
        public readonly List<GameObject> Owned=new List<GameObject>();
        public Vector3 Position(float x,float y,float z)=>Root.transform.TransformPoint(new Vector3(x,y,z));
        public Vector3 Forward=>Root.transform.forward;
        public Vector3 Up=>Root.transform.up;
        public float LocalDepth(Vector3 p)=>Root.transform.InverseTransformPoint(p).z+Cell*6;
        public Vector3 Entry(int index)=>Position((index-1)*4*Cell,.125f,-6*Cell);
        public static Material Mat(Color color) {var m=new Material(Shader.Find("Standard"));m.color=color;m.SetFloat("_Glossiness",.28f);return m;}
        Material floor,wall,metal;PhysicsMaterial contact;
        public void Build(int family,int mapSeed,bool materialOOD=false) {
            Family=family;var rng=new System.Random(mapSeed);Cell=.093f+(float)rng.NextDouble()*.012f;
            Friction=materialOOD?.55f:.10f+(float)rng.NextDouble()*.08f;
            contact=new PhysicsMaterial("PurumoguFriction"){dynamicFriction=Friction,staticFriction=Friction+.03f,bounciness=.05f,frictionCombine=PhysicsMaterialCombine.Average};
            Root=new GameObject("TruthGeometry");Owned.Add(Root);
            Root.transform.rotation=Quaternion.Euler(12f+(float)rng.NextDouble()*5,0,0);
            floor=Mat(new Color(.17f,.23f,.28f));wall=Mat(new Color(.37f,.43f,.47f));metal=Mat(new Color(.62f,.71f,.72f));
            Box("SlopedBase",new Vector3(0,-.035f,0),new Vector3(11*Cell,.07f,15*Cell),floor,false);
            var open=new bool[11,13];Action<int,int,int,int> path=(x0,z0,x1,z1)=>{int x=x0,z=z0;open[x,z]=true;while(x!=x1){x+=Math.Sign(x1-x);open[x,z]=true;}while(z!=z1){z+=Math.Sign(z1-z);open[x,z]=true;}};
            for(int e=1;e<=9;e+=4)path(e,0,e,2);path(1,2,9,2);
            if(family==0||family==4){path(5,2,5,12);}
            else if(family==1){path(5,2,5,6);path(2,6,8,6);path(2,6,2,12);path(8,6,8,12);}
            else if(family==2){path(5,2,5,7);}
            else if(family==3){path(5,2,5,4);path(3,4,7,4);path(3,4,3,9);path(7,4,7,9);path(3,9,7,9);path(5,9,5,12);}
            else{path(5,2,5,12);path(2,5,8,5);path(2,5,2,10);path(8,5,8,8);path(2,10,5,10);}
            // Runs of solid cells preserve the corridor graph without per-frame mesh work.
            for(int z=0;z<13;z++){int x=0;while(x<11){if(open[x,z]){x++;continue;}int start=x;while(x<11&&!open[x,z])x++;Box("Wall",new Vector3(((start+x-1)*.5f-5)*Cell,.065f,(z-6)*Cell),new Vector3((x-start)*Cell,.13f,Cell),wall,true);}}
            Box("BoundaryLeft",new Vector3(-5.55f*Cell,.065f,0),new Vector3(.02f,.13f,14*Cell),metal,true);
            Box("BoundaryRight",new Vector3(5.55f*Cell,.065f,0),new Vector3(.02f,.13f,14*Cell),metal,true);
            Box("EndTray",new Vector3(0,.04f,7.3f*Cell),new Vector3(11*Cell,.08f,.025f),metal,true);
            if(family==4){float gap=.048f+(float)rng.NextDouble()*.008f;float width=(Cell-gap)/2;
                foreach(int sign in new[]{-1,1})Box("Constriction",new Vector3(sign*(gap/2+width/2),.05f,1.0f*Cell),new Vector3(width,.10f,1.7f*Cell),metal,true);
            }
            Physics.SyncTransforms();
        }
        public GameObject Box(string name,Vector3 pos,Vector3 size,Material mat,bool wallLayer){var g=GameObject.CreatePrimitive(PrimitiveType.Cube);g.name=name;g.transform.SetParent(Root.transform,false);g.transform.localPosition=pos;g.transform.localScale=size;g.GetComponent<Renderer>().sharedMaterial=mat;if(wallLayer)g.layer=6;g.GetComponent<Collider>().sharedMaterial=contact;return g;}
        public void Dispose(){foreach(var o in Owned)if(o)UnityEngine.Object.DestroyImmediate(o);Owned.Clear();if(contact)UnityEngine.Object.DestroyImmediate(contact);if(floor)UnityEngine.Object.DestroyImmediate(floor);if(wall)UnityEngine.Object.DestroyImmediate(wall);if(metal)UnityEngine.Object.DestroyImmediate(metal);}
    }
}
