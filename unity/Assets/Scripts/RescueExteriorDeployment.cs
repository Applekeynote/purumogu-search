using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueScene
    {
        public struct ExteriorOpening
        {
            public Vector3 FloorPoint,Outward;
            public float HalfWall;
            public int Level;
        }
        // These are externally selected release sites, not interior occupancy or source labels.
        public readonly List<ExteriorOpening> ExteriorOpenings=new List<ExteriorOpening>();

        public void ConfigureExteriorDeployment(int worldSeed)
        {
            if(ExteriorOpenings.Count>0)return;
            var rng=new System.Random(worldSeed^0x41d607);
            var walls=Root.GetComponentsInChildren<BoxCollider>().Where(IsExteriorWall).ToArray();
            for(int level=0;level<FloorCount;level++)
            {
                float height=FloorElevation(level);var candidates=new List<ExteriorOpening>();int atHeight=0,footprintSides=0;var obscured=new List<string>();
                foreach(var wall in walls)
                {
                    var b=wall.bounds;if(Mathf.Abs(b.min.y-height)>.03f)continue;
                    atHeight++;
                    var axis=b.size.x<b.size.z?Vector3.right:Vector3.forward;
                    // Legacy side walls are long: sample several externally visible positions.
                    int samples=Mathf.Max(1,Mathf.FloorToInt(Mathf.Max(b.size.x,b.size.z)/.3f));
                    for(int k=0;k<samples;k++)
                    {
                        var point=b.center;var tangent=Vector3.Cross(Vector3.up,axis);
                        point+=tangent*((k+.5f)/samples-.5f)*Mathf.Max(b.size.x,b.size.z);point.y=height;
                        bool plus=ContainsExteriorFootprint(point+axis*.06f,level),minus=ContainsExteriorFootprint(point-axis*.06f,level);
                        if(plus==minus)continue;
                        footprintSides++;
                        var outward=plus?-axis:axis;float half=(axis.x!=0?b.size.x:b.size.z)*.5f;
                        var sight=point+Vector3.up*.15f;
                        if(Physics.Linecast(sight+outward*1.6f,sight+outward*(half+.008f),out RaycastHit obstacle,1<<6)){obscured.Add(obstacle.collider.name);continue;}
                        candidates.Add(new ExteriorOpening{FloorPoint=point,Outward=outward,HalfWall=half,Level=level});
                    }
                }
                // A collapsed facade may already have no wall. Its visible deck
                // perimeter is a valid physical release edge, without inventing a wall.
                if(ComplexGeometry)foreach(var zone in FloorZones.Where(f=>Mathf.Abs(f.Height-height)<.001f))
                {
                    var polygon=zone.Polygon??RectanglePolygon(zone.Bounds);
                    for(int edge=0;edge<polygon.Length;edge++)
                    {
                        var a=polygon[edge];var b=polygon[(edge+1)%polygon.Length];var tangent=(b-a).normalized;
                        var outward=new Vector3(tangent.y,0,-tangent.x);int samples=Mathf.Max(1,Mathf.FloorToInt((b-a).magnitude/.12f));
                        for(int k=0;k<samples;k++)
                        {
                            var p=Vector2.Lerp(a,b,(k+.5f)/samples);var point=new Vector3(p.x,height,p.y);
                            if(ContainsExteriorFootprint(point+outward*.03f,level)||!ContainsExteriorFootprint(point-outward*.03f,level))continue;
                            var existing=walls.FirstOrDefault(w=>w&&w.bounds.SqrDistance(point+Vector3.up*.10f)<.00000625f);
                            float half=existing?Vector3.Dot(new Vector3(Mathf.Abs(outward.x),0,Mathf.Abs(outward.z)),existing.bounds.size)*.5f:0;
                            var sight=point+Vector3.up*.15f;
                            if(Physics.Linecast(sight+outward*1.6f,sight+outward*(half+.008f),1<<6))continue;
                            if(candidates.Any(c=>Vector3.Distance(c.FloorPoint,point)<.04f))continue;
                            candidates.Add(new ExteriorOpening{FloorPoint=point,Outward=outward,HalfWall=half,Level=level});
                        }
                    }
                }
                if(ComplexGeometry)candidates.RemoveAll(site=>
                {
                    var tangent=Vector3.Cross(Vector3.up,site.Outward);var inside=site.FloorPoint-site.Outward*.04f;
                    return !ContainsExteriorFootprint(inside+tangent*.09f,level)||!ContainsExteriorFootprint(inside-tangent*.09f,level);
                });
                var selected=new List<ExteriorOpening>();
                while(candidates.Count>0&&selected.Count<4)
                {
                    int index=selected.Count==0?rng.Next(candidates.Count):Enumerable.Range(0,candidates.Count).OrderByDescending(i=>selected.Min(p=>Vector3.Distance(p.FloorPoint,candidates[i].FloorPoint))).First();
                    var opening=candidates[index];selected.Add(opening);
                    candidates.RemoveAll(p=>Vector3.Distance(p.FloorPoint,opening.FloorPoint)<.30f);
                }
                if(selected.Count<2)throw new InvalidOperationException("Insufficient externally accessible release openings: "+Spec.Id+" level "+level+" walls="+atHeight+" footprint="+footprintSides+" selected="+selected.Count+" occluded="+string.Join(",",obscured.Distinct())+" wall levels="+string.Join(";",walls.GroupBy(w=>w.name).Select(g=>g.Key+":"+g.Count()+" y="+g.First().bounds.min.y+" pos="+g.First().transform.position.y)));
                ExteriorOpenings.AddRange(selected);
            }
            foreach(var opening in ExteriorOpenings)if(opening.HalfWall>0){CarveExteriorOpening(opening);Physics.SyncTransforms();}
            Physics.SyncTransforms();
        }
        static bool IsExteriorWall(BoxCollider wall)=>wall.name.StartsWith("FootprintWall",StringComparison.Ordinal)||wall.name.StartsWith("BrokenSide",StringComparison.Ordinal)||wall.name.StartsWith("ExteriorWallRemnant",StringComparison.Ordinal);
        bool ContainsExteriorFootprint(Vector3 point,int level)
        {
            if(!ComplexGeometry)return Mathf.Abs(point.x)<.66f&&point.z>-.2f&&point.z<2.8f;
            return FloorZones.Any(f=>Mathf.Abs(f.Height-FloorElevation(level))<.001f&&f.Contains(new Vector2(point.x,point.z)));
        }
        void CarveExteriorOpening(ExteriorOpening opening)
        {
            bool thinX=opening.Outward.x!=0;var center=opening.FloorPoint;
            float normal=thinX?center.x:center.z,lateral=thinX?center.z:center.x;
            foreach(var wall in Root.GetComponentsInChildren<BoxCollider>().Where(IsExteriorWall).ToArray())
            {
                var b=wall.bounds;float plane=thinX?b.center.x:b.center.z;
                if(Mathf.Abs(plane-normal)>.002f||Mathf.Abs(b.min.y-center.y)>.03f)continue;
                float lo=thinX?b.min.z:b.min.x,hi=thinX?b.max.z:b.max.x;
                float left=Mathf.Max(lo,lateral-.10f),right=Mathf.Min(hi,lateral+.10f);
                if(right<=left)continue;
                var material=wall.GetComponent<Renderer>().sharedMaterial;UnityEngine.Object.DestroyImmediate(wall.gameObject);
                Action<float,float,float,float> piece=(a,z,bottom,top)=>
                {
                    if(z-a<.0005f||top-bottom<.0005f)return;
                    var p=thinX?new Vector3(plane,(bottom+top)*.5f,(a+z)*.5f):new Vector3((a+z)*.5f,(bottom+top)*.5f,plane);
                    var size=thinX?new Vector3(b.size.x,top-bottom,z-a):new Vector3(z-a,top-bottom,b.size.z);
                    WorldBox("ExteriorWallRemnant_L"+opening.Level,p,size,material);
                };
                piece(lo,left,b.min.y,b.max.y);piece(right,hi,b.min.y,b.max.y);
                piece(left,right,b.min.y,Mathf.Min(b.max.y,center.y));
                piece(left,right,Mathf.Max(b.min.y,center.y+.215f),b.max.y);
            }
        }
    }
}
