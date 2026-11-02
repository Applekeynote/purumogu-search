using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueController
    {
        FrameMeasurement performance;
        void ConfigurePerformance(string[] args)
        {
            string output=null;float seconds=45;
            for(int i=0;i<args.Length-1;i++)
            {if(args[i]=="--performance-output")output=args[i+1];if(args[i]=="--performance-seconds")float.TryParse(args[i+1],out seconds);}
            if(output!=null){performance=new FrameMeasurement(output,Mathf.Clamp(seconds,15,600),simulation.Scene.Spec.Id,simulation.Robots.Count);Begin();}
        }
        long MeasureStart()=>performance==null?0:Stopwatch.GetTimestamp();
        void MeasureEnd(string name,long start){if(performance!=null)performance.Record(name,start);}
        void LateUpdate(){if(performance!=null&&performance.Frame(simulation?.Step??0,waiting,latency,failure))Application.Quit();}
        sealed class FrameMeasurement
        {
            readonly string output,scene;readonly int nodes;readonly double seconds;readonly Stopwatch clock=Stopwatch.StartNew();
            readonly List<double> frames=new List<double>();
            readonly Dictionary<string,List<double>> stages=new Dictionary<string,List<double>>();
            double last;int waitingFrames;
            public FrameMeasurement(string output,double seconds,string scene,int nodes){this.output=output;this.seconds=seconds;this.scene=scene;this.nodes=nodes;}
            public void Record(string name,long start)
            {
                if(clock.Elapsed.TotalSeconds<5)return;
                if(!stages.TryGetValue(name,out var values)){values=new List<double>();stages[name]=values;}
                values.Add((Stopwatch.GetTimestamp()-start)*1000d/Stopwatch.Frequency);
            }
            static object Stats(List<double> values)
            {
                if(values.Count==0)return new{count=0};var sorted=values.ToArray();Array.Sort(sorted);double sum=0;int over33=0,over50=0;
                foreach(double value in sorted){sum+=value;if(value>33.333)over33++;if(value>50)over50++;}
                return new{count=sorted.Length,mean_ms=sum/sorted.Length,p50_ms=sorted[(int)((sorted.Length-1)*.5)],p95_ms=sorted[(int)((sorted.Length-1)*.95)],p99_ms=sorted[(int)((sorted.Length-1)*.99)],max_ms=sorted[sorted.Length-1],over_33ms=over33,over_50ms=over50};
            }
            public bool Frame(int step,bool waiting,float latency,string failure)
            {
                double now=clock.Elapsed.TotalSeconds;
                if(last>=5){frames.Add((now-last)*1000);if(waiting)waitingFrames++;}last=now;
                if(now<seconds)return false;
                var sections=new Dictionary<string,object>();foreach(var stage in stages)sections[stage.Key]=Stats(stage.Value);
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
                File.WriteAllText(output,JsonConvert.SerializeObject(new{unity=Application.unityVersion,scene,nodes,graphics=SystemInfo.graphicsDeviceName,processor=SystemInfo.processorType,seconds=now,warmup_seconds=5,steps=step,waiting_frames=waitingFrames,last_inference_ms=latency,failure,frame=Stats(frames),stages=sections},Formatting.Indented));
                return true;
            }
        }
    }
}
