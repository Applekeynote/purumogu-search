using UnityEngine;
namespace Purumogu {
    public sealed class GateActuator {
        public enum State{Disarmed,Opening,Dispensing,Closing,Error}
        public State Current {get;private set;}
        readonly SimulationWorld world;readonly int entry;
        readonly GameObject flap;readonly Rigidbody body;readonly Vector3 closed;
        int ticks;bool emitted;
        public GateActuator(SimulationWorld w,int e){world=w;entry=e;closed=new Vector3((e-1)*4*w.Terrain.Cell,.095f,-6*w.Terrain.Cell);flap=w.Terrain.Box("GateE"+e,closed,new Vector3(.08f,.012f,.31f),w.GateMaterial,false);body=flap.AddComponent<Rigidbody>();body.isKinematic=true;}
        public void Arm(){Current=State.Opening;ticks=0;emitted=false;}
        public void Stop(){Current=State.Disarmed;body.position=world.Terrain.Position(closed.x,closed.y,closed.z);}
        public void Tick(){if(Current==State.Disarmed||Current==State.Error)return;ticks++;
            if(Current==State.Opening){float t=Mathf.Clamp01(ticks/20f);body.MovePosition(world.Terrain.Root.transform.TransformPoint(closed+Vector3.right*(t*.105f)));if(t>=1){Current=State.Dispensing;world.Emit("GATE"+entry,"GATE_OPENED");}}
            else if(Current==State.Dispensing&&ticks>=80){Current=State.Closing;}
            else if(Current==State.Closing){float t=Mathf.Clamp01((ticks-80)/20f);body.MovePosition(world.Terrain.Root.transform.TransformPoint(closed+Vector3.right*((1-t)*.105f)));if(t>=1&&!emitted){emitted=true;world.Emit("GATE"+entry,"GATE_CLOSED");Current=State.Disarmed;}}
        }
    }
}
