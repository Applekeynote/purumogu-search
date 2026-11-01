using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueScene
    {
        // Post-collapse geometry, not a structural failure solver. These records
        // belong to simulator/audit only; they never enter RescuePacket or Rust.
        public sealed class CollapseLayer
        {
            public int Level;public float OriginalHeight,SurvivingArea,MissingArea;
            public readonly List<GameObject> FallenSlabs=new List<GameObject>();
        }
        public readonly List<CollapseLayer> CollapseLayers=new List<CollapseLayer>();

        void BuildPartialCollapse(int index,int seed,int? sourceSeed)
        {
            int kind=index-30;FloorCount=kind<2?2:3;
            floorElevations.Add(0);
            for(int level=1;level<FloorCount;level++)floorElevations.Add(floorElevations[level-1]+Range(.25f,.32f));
            int shape=new[]{-1,-1,0,6,3,4}[kind];
            var footprint=new HashSet<Vector2Int>();
            for(int x=-6;x<6;x++)for(int z=1;z<28;z++)
                if(shape<0||Shape(shape,(x+.5f)*.1f,(z+.5f)*.1f))footprint.Add(new Vector2Int(x,z));
            DeploymentRegions.AddRange(MergeTiles(footprint));
            for(int level=0;level<FloorCount;level++)
            {
                float height=FloorElevation(level);
                // Independent, coherent break locations per storey. A continuous
                // facade strip remains for genuine exterior deployment sites.
                float cx=Range(-.16f,.16f),cz=Range(1.0f,1.8f)+(level-1)*Range(-.18f,.18f);
                float width=Range(.48f,.72f),depth=Range(.65f,1.05f);
                var removed=new HashSet<Vector2Int>();
                if(level>0)foreach(var tile in footprint)
                {
                    float x=(tile.x+.5f)*.1f,z=(tile.y+.5f)*.1f;
                    bool primary=Mathf.Abs(x-cx)<width*.5f&&Mathf.Abs(z-cz)<depth*.5f;
                    bool branch=kind>=3&&Mathf.Abs(x+.25f)<.20f&&Mathf.Abs(z-(cz+.55f))<.35f;
                    if((primary||branch)&&Mathf.Abs(x)<.46f&&z>.4f&&z<2.4f)removed.Add(tile);
                }
                if(level>0&&removed.Count<3)
                {
                    // Narrow arms of nonrectangular footprints need their own
                    // interior break; select by geometry, never source positions.
                    var interior=footprint.Where(t=>new[]{Vector2Int.left,Vector2Int.right,Vector2Int.up,Vector2Int.down}.All(d=>footprint.Contains(t+d))).OrderBy(t=>Mathf.Abs((t.y+.5f)*.1f-cz)).ThenBy(t=>t.x).ToArray();
                    if(interior.Length==0)throw new InvalidOperationException("No collapse interior: "+Spec.Id);
                    var center=interior[0];foreach(var t in footprint)if(Mathf.Abs(t.x-center.x)<=1&&Mathf.Abs(t.y-center.y)<=2)removed.Add(t);
                }
                var intact=new HashSet<Vector2Int>(footprint);intact.ExceptWith(removed);
                foreach(var r in MergeTiles(intact))
                {
                    GameObject surface;
                    if(level==0)surface=WorldBox("ShapedFloor_L0",new Vector3(r.center.x,height-.018f,r.center.y),new Vector3(r.width,.036f,r.height),Ground);
                    else{Roof("ShapedFloor_L"+level,new Vector3(r.center.x,height-.018f,r.center.y)/Scale,new Vector3(r.width,.036f,r.height)/Scale,Vector3.zero);surface=Roofs[Roofs.Count-1].gameObject;}
                    FloorZones.Add(new FloorZone{Bounds=r,Height=height,Surface=surface});
                }
                foreach(var tile in footprint)foreach(var d in new[]{Vector2Int.left,Vector2Int.right,Vector2Int.up,Vector2Int.down})
                {
                    if(footprint.Contains(tile+d))continue;
                    var p=new Vector3((tile.x+.5f+d.x*.5f)*.1f,height+.105f,(tile.y+.5f+d.y*.5f)*.1f);
                    WorldBox("FootprintWall_L"+level,p,d.x!=0?new Vector3(.025f,.21f,.102f):new Vector3(.102f,.21f,.025f),Concrete);
                }
                if(level==0)continue;
                var layer=new CollapseLayer{Level=level,OriginalHeight=height,SurvivingArea=intact.Count*.01f,MissingArea=removed.Count*.01f};
                CollapseLayers.Add(layer);
                // Keep separate slab fragments and real inter-layer voids instead
                // of filling the hole with a solid visual-only rubble volume.
                foreach(var rectangle in MergeTiles(removed))
                {
                    int count=Mathf.Max(1,Mathf.CeilToInt(rectangle.height/.35f));
                    bool vee=kind==2||kind==5&&level==2;
                    for(int part=0;part<count;part++)for(int half=0;half<(vee?2:1);half++)
                    {
                        float z=rectangle.yMin+(part+.5f)*rectangle.height/count;
                        float roll=kind==0?Range(-3,3):vee?(half==0?-20:20):Range(10,23)*(level%2==0?-1:1);
                        float drop=kind==0?Range(.115f,.145f):Range(.065f,.105f);
                        float x=rectangle.center.x+(vee?(half==0?-.25f:.25f)*rectangle.width:0);
                        var p=new Vector3(x,height-drop,z);
                        var size=new Vector3(Mathf.Max(.035f,rectangle.width/(vee?2:1)-.012f),.025f,rectangle.height/count-.012f);
                        Roof("CollapsedLayerSlab_L"+level,p/Scale,size/Scale,new Vector3(kind>=3&&!vee?Range(-9,9):0,vee||rectangle.width<.3f?0:Range(-3,3),roll));
                        var slab=Roofs[Roofs.Count-1].gameObject;layer.FallenSlabs.Add(slab);
                        Physics.SyncTransforms();
                        // Broken masonry at both ends supports the retained slab.
                        // Probe real lower geometry, allowing layers to rest on
                        // an earlier fallen slab rather than an imaginary floor.
                        foreach(float sign in kind==4?new[]{roll>0?.42f:-.42f}:new[]{-.37f,.37f})
                        {
                            var underside=slab.transform.TransformPoint(new Vector3(sign,-.5f,0));
                            if(!Physics.Raycast(underside-Vector3.up*.002f,Vector3.down,out RaycastHit hit,.7f,1<<6))throw new InvalidOperationException("Unsupported collapse slab: "+Spec.Id+" level="+level+" position="+underside.ToString("F4"));
                            float gap=underside.y-hit.point.y;
                            if(gap>.004f)WorldBox("LayerBearingRubble_L"+level,new Vector3(underside.x,hit.point.y+gap*.5f,underside.z),new Vector3(.035f,gap,.05f),Stone);
                        }
                    }
                }
            }
            // Fragments on every retained storey create occlusion and contact.
            foreach(var zone in FloorZones)
            {
                int pieces=Mathf.CeilToInt(zone.Bounds.width*zone.Bounds.height*(20+Spec.Difficulty*10));
                for(int i=0;i<pieces;i++)
                {
                    var size=new Vector3(Range(.025f,.085f),Range(.015f,.045f),Range(.035f,.10f));
                    var p=new Vector3(Range(zone.Bounds.xMin+.01f,zone.Bounds.xMax-.01f),zone.Height+size.y*.35f,Range(zone.Bounds.yMin+.01f,zone.Bounds.yMax-.01f));
                    var fragment=WorldBox("LayerDebris",p,size,i%3==0?Stone:Concrete);Fracture(fragment);fragment.transform.rotation=Quaternion.Euler(Range(-12,12),Range(0,180),Range(-15,15));
                }
            }
            if(WaterHeight>0)foreach(var zone in FloorZones.Where(f=>f.Height<.01f))
            {
                var water=WorldBox("ResidualWaterSurface",new Vector3(zone.Bounds.center.x,WaterHeight,zone.Bounds.center.y),new Vector3(zone.Bounds.width,.002f,zone.Bounds.height),Water);UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());
            }
            Physics.SyncTransforms();
            var rng=new System.Random((sourceSeed??seed)^0x4f1bbcdc^(index*397));
            if(kind==3||kind==5)GasLeakPosition=SampleComplexPocket(new System.Random(seed^0x3511a9),-1);
            for(int i=0;i<Spec.Victims;i++)
            {
                var p=SampleComplexPocket(rng,i%FloorCount);VictimPositions.Add(p);
                Sources.Add(new AcousticSource{Position=p,Human=true,VoiceBand=.84f,Modulation=.82f,Power=1,Warmth=3.6f});
                var marker=GameObject.CreatePrimitive(PrimitiveType.Sphere);marker.name="EvaluationOnlyPerson";marker.transform.SetParent(Root.transform,false);marker.transform.position=p;marker.transform.localScale=Vector3.one*.045f;UnityEngine.Object.DestroyImmediate(marker.GetComponent<Collider>());
                var beacon=new Material(Resources.Load<Shader>("RescueTruthBeacon"));resources.Add(beacon);var renderer=marker.GetComponent<Renderer>();renderer.sharedMaterial=beacon;renderer.enabled=false;VictimRenderers.Add(renderer);
            }
            Sources.Add(new AcousticSource{Position=SampleComplexPocket(rng,-1),VoiceBand=.25f,Modulation=.28f,Power=.55f});
            Physics.SyncTransforms();Cutaway(true);
        }
    }
}
