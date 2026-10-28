using UnityEngine;

namespace Purumogu.Rescue
{
    /// Independent, bounded interaction for a single render viewport.
    public sealed class RescueViewport
    {
        Vector2 target,angles,velocity;
        float zoom=1,targetZoom=1,zoomVelocity;
        int capture;
        public void Input(Rect area,bool blocked,int hint)
        {
            var e=Event.current;int id=GUIUtility.GetControlID(hint,FocusType.Passive);
            if(e.type==EventType.ScrollWheel&&!blocked&&area.Contains(e.mousePosition)&&GUIUtility.hotControl==0)
            {targetZoom=Mathf.Clamp(targetZoom*Mathf.Exp(e.delta.y*.035f),.38f,2.5f);e.Use();}
            else if(e.type==EventType.MouseDown&&(e.button==0||e.button==2)&&!blocked&&area.Contains(e.mousePosition)&&GUIUtility.hotControl==0)
            {GUIUtility.hotControl=id;capture=id;e.Use();}
            else if(e.type==EventType.MouseDrag&&capture==id&&GUIUtility.hotControl==id)
            {target.x=Mathf.Repeat(target.x+e.delta.x*.16f+180,360)-180;target.y=Mathf.Clamp(target.y+e.delta.y*.16f,-22,42);e.Use();}
            else if(e.rawType==EventType.MouseUp&&capture==id&&GUIUtility.hotControl==id)
            {capture=0;GUIUtility.hotControl=0;e.Use();}
        }
        public void Step(float dt)
        {
            dt=Mathf.Clamp(dt,.0001f,.05f);
            angles.x=Mathf.SmoothDampAngle(angles.x,target.x,ref velocity.x,.075f,60,dt);
            angles.y=Mathf.SmoothDampAngle(angles.y,target.y,ref velocity.y,.075f,60,dt);
            zoom=Mathf.SmoothDamp(zoom,targetZoom,ref zoomVelocity,.10f,3,dt);
        }
        public void Apply(Camera camera,Vector3 focus,Vector3 initialOffset,float baseSize)
        {
            float yaw=Mathf.Atan2(initialOffset.x,initialOffset.z)*Mathf.Rad2Deg+angles.x;
            float pitch=Mathf.Clamp(Mathf.Atan2(initialOffset.y,new Vector2(initialOffset.x,initialOffset.z).magnitude)*Mathf.Rad2Deg+angles.y,10,82)*Mathf.Deg2Rad;
            var direction=new Vector3(Mathf.Sin(yaw*Mathf.Deg2Rad)*Mathf.Cos(pitch),Mathf.Sin(pitch),Mathf.Cos(yaw*Mathf.Deg2Rad)*Mathf.Cos(pitch));
            camera.transform.position=focus+direction*initialOffset.magnitude;camera.transform.LookAt(focus);camera.orthographicSize=baseSize*zoom;
        }
        public void Reset(){target=Vector2.zero;targetZoom=1;}
        public void ResetZoom(){targetZoom=1;}
        public void Cancel(){if(capture!=0&&GUIUtility.hotControl==capture)GUIUtility.hotControl=0;capture=0;}
    }
}
