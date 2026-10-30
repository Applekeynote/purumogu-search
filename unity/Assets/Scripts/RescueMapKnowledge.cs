using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Purumogu.Rescue
{
    public enum RescueMapProjection { Overview, Top, Longitudinal, Transverse }
    public enum RescueEvidenceTier { Unknown, Measured, HighEstimate, LowEstimate }

    /// Display-only evidence classification. Never reads simulation geometry or people.
    public sealed class RescueMapKnowledge : System.IDisposable
    {
        public const int TotalCells=32*21*70;
        public sealed class Cell
        {
            public Vector3Int Key;
            public int Observations;
            public float Sensor,Occupancy;
            public RescueEvidenceTier Tier;
            public bool Solid=>Tier==RescueEvidenceTier.Measured?Sensor>.6f:Occupancy>.6f;
        }
        public readonly Dictionary<Vector3Int,Cell> Cells=new Dictionary<Vector3Int,Cell>();
        public readonly List<Vector3> Frontiers=new List<Vector3>();
        public int Observed,Confirmed,High,Low,Unknown;
        public float CellSize{get;private set;}=.05f;
        public float ObservedFraction=>Observed/(float)TotalCells;
        public float InferredFraction=>(High+Low)/(float)TotalCells;
        public float UnknownFraction=>Unknown/(float)TotalCells;
        public Texture2D Section{get;private set;}
        public static bool InDomain(Vector3Int k)=>k.x>=-16&&k.x<16&&k.y>=-3&&k.y<18&&k.z>=-6&&k.z<64;
        public Vector3 Center(Vector3Int k)=>(new Vector3(k.x,k.y,k.z)+Vector3.one*.5f)*CellSize;
        public static readonly Color ObservedColor=new Color(.38f,.84f,.76f),HighColor=new Color(.36f,.67f,.73f),LowColor=new Color(.27f,.38f,.43f),UnknownColor=new Color(.038f,.060f,.077f);
        public void Update(JObject reply)
        {
            CellSize=(float)reply["cell_m"];Cells.Clear();Frontiers.Clear();Observed=Confirmed=High=Low=0;
            var measured=new HashSet<Vector3Int>();
            foreach(var token in (JArray)reply["cells"])
            {
                var a=token["key"];var key=new Vector3Int((int)a[0],(int)a[1],(int)a[2]);if(!InDomain(key))continue;
                int count=(int)token["observations"];float sensor=(float?)token["sensor_occupancy"]??.5f;
                var c=new Cell{Key=key,Observations=count,Sensor=sensor,Occupancy=(float)token["occupancy"]};
                if(count>=2&&(sensor<=.4f||sensor>=.6f)){c.Tier=RescueEvidenceTier.Measured;measured.Add(key);Confirmed++;}
                else c.Tier=count>0?RescueEvidenceTier.LowEstimate:RescueEvidenceTier.Unknown;
                if(count>0)Observed++;Cells[key]=c;
            }
            // A family softmax is not calibrated cell confidence: local measured support is mandatory.
            var supports=new Dictionary<Vector3Int,int>();
            foreach(var candidate in Cells.Values)
            {
                if(candidate.Observations>0)continue;int count=0;
                for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++)
                {
                    var n=candidate.Key+new Vector3Int(x,y,z);
                    if(measured.Contains(n)&&(candidate.Occupancy>.6f)==(Cells[n].Sensor>.6f))count++;
                }
                supports[candidate.Key]=count;
            }
            foreach(var c in Cells.Values)
            {
                if(c.Observations>0)continue;
                supports.TryGetValue(c.Key,out int count);
                // Only reveal completion next to observations, never the whole learned template.
                if((long)reply["sim_step"]>=500&&count>0)
                {
                    bool high=count>=3&&c.Occupancy>=.85f&&((float?)reply["completion_confidence"]??0)>=.8f;
                    c.Tier=high?RescueEvidenceTier.HighEstimate:RescueEvidenceTier.LowEstimate;
                    if(high)High++;else Low++;
                }
            }
            Unknown=TotalCells-Observed-High-Low;
            if(reply["goals"] is JArray goals)foreach(var goal in goals)
                if(((string)goal["reason"]??"").Contains("frontier"))
                {var p=RescueEvaluation.Vector(goal["position"]);if(Frontiers.TrueForAll(q=>Vector3.Distance(p,q)>.25f))Frontiers.Add(p);}
        }
        public bool InSlice(Vector3 p,RescueMapProjection mode,float position)
        {return mode==RescueMapProjection.Overview||Mathf.Abs((mode==RescueMapProjection.Longitudinal?p.x:mode==RescueMapProjection.Top?p.y:p.z)-position)<=CellSize*.51f;}
        public bool HasObservedCavity(Vector3 p)
        {
            int found=0;foreach(var c in Cells.Values)if(c.Tier==RescueEvidenceTier.Measured&&!c.Solid&&Vector3.Distance(Center(c.Key),p)<.15f)found++;
            return found>=8; // Local free volume only. Does not claim connected or traversable passage.
        }
        public bool HasApproachGoal(JObject reply,Vector3 p)
        {
            if(reply["goals"] is JArray goals)foreach(var g in goals)if(Vector3.Distance(RescueEvaluation.Vector(g["position"]),p)<.25f)return true;
            return false;
        }
        public Texture2D BuildSection(RescueMapProjection mode,float position,bool completion)
        {
            bool transverse=mode==RescueMapProjection.Transverse,top=mode==RescueMapProjection.Top;
            int width=transverse?32:70,height=top?32:21;
            if(Section==null||Section.width!=width*4||Section.height!=height*4)
            {if(Section)Object.DestroyImmediate(Section);Section=new Texture2D(width*4,height*4,TextureFormat.RGBA32,false){filterMode=FilterMode.Point};}
            var pixels=new Color[Section.width*Section.height];
            for(int i=0;i<pixels.Length;i++)pixels[i]=UnknownColor;
            for(int a=0;a<width;a++)for(int b=0;b<height;b++)
            {
                var k=top?new Vector3Int(b-16,Mathf.FloorToInt(position/CellSize),a-6):transverse?new Vector3Int(a-16,b-3,Mathf.FloorToInt(position/CellSize)):new Vector3Int(Mathf.FloorToInt(position/CellSize),b-3,a-6);
                Cells.TryGetValue(k,out var c);bool visible=c!=null&&c.Tier!=RescueEvidenceTier.Unknown&&(completion||c.Tier==RescueEvidenceTier.Measured);
                for(int x=0;x<4;x++)for(int y=0;y<4;y++)
                {
                    Color color=UnknownColor;
                    if(!visible){if((a*4+x+b*4+y)%10==0)color=new Color(.075f,.1f,.12f);}
                    else if(c.Tier==RescueEvidenceTier.Measured)
                        color=c.Solid?(x==0||y==0?new Color(.75f,.95f,.95f):ObservedColor):new Color(.09f,.23f,.26f);
                    else if(c.Tier==RescueEvidenceTier.HighEstimate)color=(x+y)%3==0?HighColor:new Color(.07f,.16f,.19f);
                    else color=(x==1&&y==1)?LowColor:UnknownColor;
                    pixels[(b*4+y)*Section.width+a*4+x]=color;
                }
            }
            Section.SetPixels(pixels);Section.Apply();return Section;
        }
        public void Dispose(){if(Section)Object.DestroyImmediate(Section);}
    }
}
