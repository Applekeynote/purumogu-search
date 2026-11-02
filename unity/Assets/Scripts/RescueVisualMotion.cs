using System.Collections.Generic;
using UnityEngine;

namespace Purumogu.Rescue
{
    /// Interpolate the last two simulated poses, without extrapolation or physics edits.
    public sealed class RescueVisualMotion
    {
        struct Pose
        {
            public Rigidbody Body;public Transform Visual;
            public Vector3 Previous,Current;public Quaternion Before,Rotation;
        }
        readonly Pose[] poses;float capturedAt;
        public RescueVisualMotion(IReadOnlyList<RescueRobot> robots)
        {
            poses=new Pose[robots.Count*3];int i=0;
            foreach(var robot in robots)foreach(var body in new[]{robot.Core,robot.Cones[0],robot.Cones[1]})
            {
                poses[i]=new Pose{Body=body,Visual=robot.Visuals[i%3],Previous=body.position,Current=body.position,Before=body.rotation,Rotation=body.rotation};i++;
            }
            capturedAt=Time.realtimeSinceStartup;
        }
        public void Capture()
        {
            for(int i=0;i<poses.Length;i++)
            {poses[i].Previous=poses[i].Current;poses[i].Before=poses[i].Rotation;poses[i].Current=poses[i].Body.position;poses[i].Rotation=poses[i].Body.rotation;}
            capturedAt=Time.realtimeSinceStartup;
        }
        public void Present(float speed)
        {PresentFraction(Mathf.Clamp01((Time.realtimeSinceStartup-capturedAt)*Mathf.Max(.1f,speed)/.02f));}
        public void PresentFraction(float fraction)
        {
            fraction=Mathf.Clamp01(fraction);
            foreach(var pose in poses)pose.Visual.SetPositionAndRotation(Vector3.Lerp(pose.Previous,pose.Current,fraction),Quaternion.Slerp(pose.Before,pose.Rotation,fraction));
        }
    }
}
