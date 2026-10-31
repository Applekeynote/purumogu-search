using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu.Rescue
{
    public enum RescueDeploymentMode { EntranceGrid, WholeSceneDrop }

    /// Release plans are known actuator inputs. They never query hidden cavities or sources.
    public static class RescueDeployment
    {
        public const int MaximumNodes=48;
        public const float DropHeight=1.05f;
        public const int ScatterSteps=600;
        public struct ReleasePose {public Vector3 Position;public Quaternion Rotation;public int Step;}

        public static List<ReleasePose> Plan(int count,int seed,RescueDeploymentMode mode,IReadOnlyList<Rect> regions=null)
        {
            if(count<1||count>MaximumNodes)throw new ArgumentOutOfRangeException(nameof(count));
            var result=new List<ReleasePose>();
            if(mode==RescueDeploymentMode.EntranceGrid)
            {
                int columns=Math.Min(12,Math.Max(3,(count+3)/4));
                for(int i=0;i<count;i++)result.Add(new ReleasePose{Position=new Vector3((i%columns-(columns-1)*.5f)*.095f,.018f,-.18f+i/columns*.074f),Rotation=Quaternion.identity,Step=0});
                return result;
            }
            if(regions!=null&&regions.Count>0)
            {
                var rng=new System.Random(seed);
                for(int i=0;i<count;i++)
                {
                    bool found=false;Vector3 position=Vector3.zero;
                    for(int attempt=0;attempt<10000;attempt++)
                    {
                        var r=regions[(i+attempt)%regions.Count];
                        float mx=Mathf.Min(.055f,r.width*.35f),mz=Mathf.Min(.055f,r.height*.35f);
                        position=new Vector3(Mathf.Lerp(r.xMin+mx,r.xMax-mx,(float)rng.NextDouble()),DropHeight,Mathf.Lerp(r.yMin+mz,r.yMax-mz,(float)rng.NextDouble()));
                        found=result.TrueForAll(other=>Vector3.Distance(position,other.Position)>=.085f);if(found)break;
                    }
                    if(!found)throw new InvalidOperationException("Scatter envelope cannot fit separated nodes");
                    result.Add(new ReleasePose{Position=position,Rotation=Quaternion.Euler(0,(float)rng.NextDouble()*360,0),Step=(int)Math.Round(i*(double)ScatterSteps/Math.Max(1,count-1))});
                }
                return result;
            }
            var random=new System.Random(seed);const int across=6;int rows=(count+across-1)/across;
            for(int i=0;i<count;i++)
            {
                Vector3 position=Vector3.zero;bool found=false;
                for(int attempt=0;attempt<2000;attempt++)
                {
                    float x=-.57f+(i%across+.5f+((float)random.NextDouble()-.5f)*.72f)*(1.14f/across);
                    float z=.1f+(i/across+.5f+((float)random.NextDouble()-.5f)*.72f)*(2.6f/rows);
                    position=new Vector3(x,DropHeight,z);found=true;
                    foreach(var other in result)if(Vector3.Distance(position,other.Position)<.085f){found=false;break;}
                    if(found)break;
                }
                if(!found)throw new InvalidOperationException("Could not make a separated whole-scene release plan");
                result.Add(new ReleasePose{Position=position,Rotation=Quaternion.Euler((float)random.NextDouble()*24-12,(float)random.NextDouble()*360,(float)random.NextDouble()*24-12)});
            }
            for(int i=result.Count-1;i>0;i--){int j=random.Next(i+1);var temporary=result[i];result[i]=result[j];result[j]=temporary;}
            for(int i=0;i<result.Count;i++){var pose=result[i];pose.Step=(int)Math.Round(i*(double)ScatterSteps/Math.Max(1,count-1));result[i]=pose;}
            return result;
        }
    }
}
