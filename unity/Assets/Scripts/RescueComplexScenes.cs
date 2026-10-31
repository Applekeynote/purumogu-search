using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueScene
    {
        public sealed class FloorZone {public Rect Bounds;public float Height;}
        public readonly List<FloorZone> FloorZones=new List<FloorZone>();
        /// Known exterior scatter envelope; no obstacles, cavities, or source positions.
        public readonly List<Rect> DeploymentRegions=new List<Rect>();
        public bool ComplexGeometry=>Spec.Layout>=15;
        public int FloorCount{get;private set;}=1;
        GameObject WorldBox(string name,Vector3 p,Vector3 size,Material material)=>Box(name,p/Scale,size/Scale,material);
        static bool RectContains(float x,float z,float left,float bottom,float width,float depth)=>x>=left&&x<left+width&&z>=bottom&&z<bottom+depth;
        static bool Shape(int shape,float x,float z)
        {
            bool core=RectContains(x,z,-.65f,.05f,1.3f,2.8f);if(!core)return false;
            switch(shape%10)
            {
                case 0:return x<-.15f||z>2.15f; // L
                case 1:return Mathf.Abs(x)<.25f||z>2.10f; // T
                case 2:return Mathf.Abs(x)<.22f&&z<1.5f||z>=1.1f&&Mathf.Abs(Mathf.Abs(x)-(z-1.05f)*.32f)<.23f; // Y
                case 3:return Mathf.Abs(x)>.25f||z<.65f; // U
                case 4:return Mathf.Abs(x)>.35f||z<.55f||z>2.35f; // courtyard
                case 5:return z<.7f||z>2.15f||Mathf.Abs(x)<.22f; // H
                case 6:return Mathf.Abs(x)<.24f||z>1.10f&&z<1.75f; // cross
                case 7:return Mathf.Abs(x)>.35f||z<.5f||z>2.4f||z>1.25f&&z<1.55f; // ring + chord
                case 8:return z<1?x<.20f:z<1.9f?x>-.20f:x<.2f; // dogleg
                default:return z<.65f||z>2.25f||Mathf.Abs(x)<.18f||x<-.3f&&z<1.65f||x>.3f&&z>1.15f;
            }
        }
        static List<Rect> MergeTiles(HashSet<Vector2Int> source)
        {
            var remaining=new HashSet<Vector2Int>(source);var result=new List<Rect>();
            foreach(var start in source.OrderBy(k=>k.y).ThenBy(k=>k.x))
            {
                if(!remaining.Contains(start))continue;int width=1,height=1;
                while(remaining.Contains(start+new Vector2Int(width,0)))width++;
                while(Enumerable.Range(0,width).All(x=>remaining.Contains(start+new Vector2Int(x,height))))height++;
                for(int x=0;x<width;x++)for(int z=0;z<height;z++)remaining.Remove(start+new Vector2Int(x,z));
                result.Add(new Rect(start.x*.1f,start.y*.1f,width*.1f,height*.1f));
            }
            return result;
        }
        void BuildComplex(int index,int seed,int? sourceSeed)
        {
            int variant=index-15;FloorCount=variant<5?1:variant<10?2:3;
            var envelope=new HashSet<Vector2Int>();
            for(int level=0;level<FloorCount;level++)
            {
                float height=level*.28f;int shape=variant%10;var tiles=new HashSet<Vector2Int>();
                for(int x=-7;x<7;x++)for(int z=0;z<29;z++)
                {
                    float px=(x+.5f)*.1f,pz=(z+.5f)*.1f;
                    if(!Shape(shape,px,pz))continue;
                    // Open shafts and staggered decks let dispersed nodes reach different storeys.
                    if(level>0&&(RectContains(px,pz,-.22f,.65f,.44f,.55f)||level==2&&pz<1.1f))continue;
                    if(level>0&&variant%3==0&&px<-.25f&&pz<1.7f)continue;
                    tiles.Add(new Vector2Int(x,z));envelope.Add(new Vector2Int(x,z));
                }
                foreach(var rectangle in MergeTiles(tiles))
                {
                    var center=new Vector3(rectangle.center.x,height-.018f,rectangle.center.y);var size=new Vector3(rectangle.width,.036f,rectangle.height);
                    if(level>0)Roof("ShapedFloor_L"+level,center/Scale,size/Scale,Vector3.zero);else WorldBox("ShapedFloor_L"+level,center,size,Ground);
                    FloorZones.Add(new FloorZone{Bounds=rectangle,Height=height});
                }
                // Boundary follows the footprint; there is no enclosing rectangular tray.
                foreach(var tile in tiles)foreach(var direction in new[]{Vector2Int.left,Vector2Int.right,Vector2Int.up,Vector2Int.down})
                {
                    if(tiles.Contains(tile+direction))continue;
                    bool xWall=direction.x!=0;var p=new Vector3((tile.x+.5f+direction.x*.5f)*.1f,height+.115f,(tile.y+.5f+direction.y*.5f)*.1f);
                    WorldBox("FootprintWall_L"+level,p,xWall?new Vector3(.025f,.23f,.102f):new Vector3(.102f,.23f,.025f),Concrete);
                }
                foreach(var zone in FloorZones.FindAll(f=>Mathf.Abs(f.Height-height)<.001f))
                {
                    int pieces=Mathf.CeilToInt(zone.Bounds.width*zone.Bounds.height*(18+Spec.Difficulty*10));
                    for(int i=0;i<pieces;i++)
                    {
                        var size=new Vector3(Range(.035f,.11f),Range(.018f,.065f),Range(.04f,.14f));
                        float x=Range(zone.Bounds.xMin+.02f,zone.Bounds.xMax-.02f),z=Range(zone.Bounds.yMin+.02f,zone.Bounds.yMax-.02f);
                        var debris=WorldBox("LevelDebris_L"+level,new Vector3(x,height+size.y*.35f,z),size,i%3==0?Stone:Concrete);
                        Fracture(debris);debris.transform.rotation=Quaternion.Euler(Range(-15,15),Range(0,180),Range(-18,18));
                    }
                }
            }
            DeploymentRegions.AddRange(MergeTiles(envelope));
            for(int level=0;level<FloorCount-1;level++)
            {
                // Each flight connects real floor zones; clear openings replace wall segments along it.
                var low=FloorZones.Where(f=>Mathf.Abs(f.Height-level*.28f)<.001f).OrderByDescending(f=>f.Bounds.width*f.Bounds.height).First();
                var high=FloorZones.Where(f=>Mathf.Abs(f.Height-(level+1)*.28f)<.001f).OrderByDescending(f=>f.Bounds.width*f.Bounds.height).First();
                Vector3 start=new Vector3(low.Bounds.center.x,low.Height+.006f,Mathf.Clamp(low.Bounds.center.y-.45f,.25f,2.25f));
                Vector3 end=new Vector3(high.Bounds.center.x,high.Height+.006f,Mathf.Clamp(high.Bounds.center.y+.35f,.65f,2.65f));
                if(new Vector2(end.x-start.x,end.z-start.z).magnitude<.65f)start.z=Mathf.Max(.15f,end.z-.85f);
                var delta=end-start;var ramp=WorldBox("ConnectingRamp_L"+level,(start+end)*.5f,new Vector3(.24f,.018f,delta.magnitude),Concrete);ramp.transform.rotation=Quaternion.LookRotation(delta,Vector3.up);
                // Door cuts exist in geometry, not just in the rendered wall material.
                foreach(var collider in Root.GetComponentsInChildren<BoxCollider>())
                {
                    if(!collider.name.StartsWith("FootprintWall"))continue;
                    if(Physics.ComputePenetration(ramp.GetComponent<Collider>(),ramp.transform.position,ramp.transform.rotation,collider,collider.transform.position,collider.transform.rotation,out _,out _))UnityEngine.Object.DestroyImmediate(collider.gameObject);
                }
                if(variant==11)for(int step=0;step<6;step++)WorldBox("StairTread",Vector3.Lerp(start,end,(step+.5f)/6),new Vector3(.12f,.035f,delta.magnitude/6),Stone);
            }
            if(WaterHeight>0)foreach(var zone in FloorZones.FindAll(f=>f.Height<.01f))
            {
                var water=WorldBox("ResidualWaterSurface",new Vector3(zone.Bounds.center.x,WaterHeight,zone.Bounds.center.y),new Vector3(zone.Bounds.width,.002f,zone.Bounds.height),Water);UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());
            }
            Physics.SyncTransforms();var rng=new System.Random((sourceSeed??seed)^0x4f1bbcdc^(index*397));
            if(variant%5==4||WaterHeight>0)GasLeakPosition=SampleComplexPocket(new System.Random(seed^0x3511a9),-1);
            for(int i=0;i<Spec.Victims;i++)
            {
                var p=SampleComplexPocket(rng,FloorCount>1?i%FloorCount:-1);VictimPositions.Add(p);
                Sources.Add(new AcousticSource{Position=p,Human=true,VoiceBand=.84f,Modulation=.82f,Power=1,Warmth=3.6f});
                var marker=GameObject.CreatePrimitive(PrimitiveType.Sphere);marker.name="EvaluationOnlyPerson";marker.transform.SetParent(Root.transform,false);marker.transform.position=p;marker.transform.localScale=Vector3.one*.045f;UnityEngine.Object.DestroyImmediate(marker.GetComponent<Collider>());
                var beacon=new Material(Resources.Load<Shader>("RescueTruthBeacon"));resources.Add(beacon);var renderer=marker.GetComponent<Renderer>();renderer.sharedMaterial=beacon;renderer.enabled=false;VictimRenderers.Add(renderer);
            }
            Sources.Add(new AcousticSource{Position=SampleComplexPocket(rng,-1),VoiceBand=.25f,Modulation=.28f,Power=.55f});
            Physics.SyncTransforms();Cutaway(true);
        }
        Vector3 SampleComplexPocket(System.Random rng,int floor)
        {
            var choices=FloorZones.FindAll(f=>floor<0||Mathf.Abs(f.Height-floor*.28f)<.001f);
            for(int attempt=0;attempt<2048;attempt++)
            {
                var zone=choices[rng.Next(choices.Count)];var r=zone.Bounds;
                var p=new Vector3(Mathf.Lerp(r.xMin+.025f,r.xMax-.025f,(float)rng.NextDouble()),zone.Height+.035f+(float)rng.NextDouble()*.045f,Mathf.Lerp(r.yMin+.025f,r.yMax-.025f,(float)rng.NextDouble()));
                if(Physics.CheckSphere(p,.012f,1<<6)||VictimPositions.Exists(other=>Vector3.Distance(other,p)<.28f))continue;return p;
            }
            throw new InvalidOperationException("No source pocket in "+Spec.Id);
        }
    }
}
