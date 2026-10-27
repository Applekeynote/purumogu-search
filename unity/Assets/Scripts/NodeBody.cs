using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu {
    public enum NodeType { Lite,Mapping,HumanCue,Tracer }
    public sealed class NodeBody:MonoBehaviour {
        public string Id {get;private set;}
        public NodeType Type=NodeType.Lite;
        public Rigidbody Body {get;private set;}
        public string Mode="CLUSTER";
        public float Energy=1;
        public bool Powered=true;
        SimulationWorld world;Vector3 previousVelocity,previousAccel;
        int quietSteps,lastImpact=-100,lastRadio,nextCoupleStep;bool still;
        System.Random rng,lossRng;double bias;long clockOffset;
        FixedJoint link;NodeBody partner;LineRenderer line;
        readonly Dictionary<string,int> lastHeard=new Dictionary<string,int>();
        public static NodeBody Create(SimulationWorld world,string id,Vector3 position,int seed){
            var g=new GameObject(id);g.transform.position=position;g.transform.rotation=world.Terrain.Root.transform.rotation;
            var b=g.AddComponent<NodeBody>();b.world=world;b.Id=id;b.rng=new System.Random(seed);b.lossRng=new System.Random(seed^0x4b31);b.bias=(b.rng.NextDouble()-.5)*.3;b.clockOffset=b.rng.Next(-30,31);
            var body=GameObject.CreatePrimitive(PrimitiveType.Cube);body.name="RoundedCore";body.transform.SetParent(g.transform,false);body.transform.localScale=new Vector3(.036f,.019f,.042f);body.GetComponent<Renderer>().sharedMaterial=world.NodeMaterial;
            body.GetComponent<Renderer>().enabled=false;NodeVisual.Attach(g.transform,world.NodeMaterial,world.ArmMaterial);
            // Four short rounded capture arms; original abstract design, no movie assets.
            foreach(Vector3 side in new[]{Vector3.left,Vector3.right,Vector3.forward,Vector3.back}){var arm=GameObject.CreatePrimitive(PrimitiveType.Capsule);arm.name="CaptureArm";arm.transform.SetParent(g.transform,false);arm.transform.localPosition=side*.021f;arm.transform.localRotation=Quaternion.FromToRotation(Vector3.up,side);arm.transform.localScale=new Vector3(.006f,.008f,.006f);arm.GetComponent<Renderer>().sharedMaterial=world.ArmMaterial;}
            foreach(var collider in g.GetComponentsInChildren<Collider>())collider.sharedMaterial=world.NodePhysics;
            b.Body=g.AddComponent<Rigidbody>();b.Body.mass=.024f;b.Body.collisionDetectionMode=CollisionDetectionMode.ContinuousDynamic;b.Body.linearDamping=.15f;b.Body.angularDamping=.8f;b.Body.solverIterations=10;b.Body.maxAngularVelocity=10;
            b.line=g.AddComponent<LineRenderer>();b.line.positionCount=0;b.line.startWidth=b.line.endWidth=.004f;b.line.sharedMaterial=world.LinkMaterial;
            world.Emit(id,"NODE_SPAWN");world.Emit(id,"NODE_DEPLOYED");return b;
        }
        public void Command(string mode,bool powered){if(Mode!=mode)world.Emit(Id,"PARENT_CMD",mode=="CHAIN_INSERT"?1:0);Mode=mode;Powered=powered;if(mode=="CLUSTER")Uncouple();world.Emit(Id,powered?"THRUST_ON":"THRUST_OFF");}
        public void MotorStep(){
            if(!Powered||Energy<=0)return;
            Vector3 heading=world.Terrain.Forward;
            // A short local feeler selects a wall-following turn, without a map or a route.
            Vector3 nose=Body.position+world.Terrain.Up*.008f;
            if(Physics.Raycast(nose,heading,.055f,1<<6)){
                Vector3 side=world.Terrain.Root.transform.right;
                bool left=Physics.Raycast(nose,-side,.07f,1<<6),right=Physics.Raycast(nose,side,.07f,1<<6);
                if(!left||!right)heading=(!right&&(!left?(Id[Id.Length-1]%2==0):true))?side:-side;
                else heading=-heading;
            }
            float force=Mode=="CHAIN_INSERT"?.060f:.045f;
            Body.AddForce(heading*force,ForceMode.Force);
            if(Mode=="CHAIN_INSERT"){
                Vector3 forward=Vector3.ProjectOnPlane(transform.forward,world.Terrain.Up).normalized;
                Body.AddTorque(Vector3.Cross(forward,heading)*.00012f-Body.angularVelocity*.000008f);
                if(!link&&world.Step>=nextCoupleStep){foreach(var other in world.Nodes){if(other==this||other.Mode!=Mode||!other.Body)continue;var delta=other.Body.position-Body.position;if(delta.magnitude<.066f&&delta.magnitude>.025f&&Vector3.Dot(delta.normalized,heading)>.65f&&Quaternion.Angle(transform.rotation,other.transform.rotation)<45&&other.IncomingLinks()<1&&IncomingLinks()<1){link=gameObject.AddComponent<FixedJoint>();link.connectedBody=other.Body;link.breakForce=.30f;link.breakTorque=.05f;link.enableCollision=false;partner=other;world.Emit(Id,"COUPLE_START",1,other.Id);break;}}}
            }
            Energy=Mathf.Max(0,Energy-.000004f);
        }
        int IncomingLinks(){int count=0;foreach(var n in world.Nodes)if(n.partner==this&&n.link)count++;return count;}
        void Uncouple(){if(link){world.Emit(Id,"COUPLE_END",0,partner?partner.Id:"");DestroyImmediate(link);}link=null;partner=null;line.positionCount=0;}
        public void Sample(){
            Vector3 acceleration=transform.InverseTransformDirection((Body.linearVelocity-previousVelocity)/SimulationWorld.Dt-Physics.gravity);previousVelocity=Body.linearVelocity;
            acceleration+=new Vector3(Noise(.18f),Noise(.18f)+(float)bias,Noise(.18f));acceleration=Vector3.ClampMagnitude(acceleration,80);
            float delta=(acceleration-previousAccel).magnitude;previousAccel=acceleration;
            float impact=Mathf.Abs(acceleration.magnitude-9.81f);
            if(impact>5f&&world.Step-lastImpact>=10){lastImpact=world.Step;world.Emit(Id,"IMPACT",Math.Round(impact*2)/2);}
            bool quiet=delta<1.2f&&Body.angularVelocity.magnitude<.25f;quietSteps=quiet?quietSteps+1:0;
            if(!still&&quietSteps>=25){still=true;world.Emit(Id,"STILL_START",0,"",.65);}
            if(still&&!quiet){still=false;world.Emit(Id,"STILL_END",0,"",.65);}
            if(world.Step-lastRadio>=10){lastRadio=world.Step;Radio();}
            if(partner&&!link){world.Emit(Id,"COUPLE_END",0,partner.Id);partner=null;line.positionCount=0;nextCoupleStep=world.Step+25;}
            if(link&&partner){line.positionCount=2;line.SetPosition(0,transform.position);line.SetPosition(1,partner.transform.position);}
        }
        void Radio(){
            // Capture-health loss is observable; an unreceived peer's identity is not.
            // Loss is correlated across the receiver's scan, not independent per link.
            if(lossRng.NextDouble()<world.PacketLoss){world.Emit(Id,"OBSERVATION_GAP",0,"",.3);return;}
            foreach(var peer in world.Nodes){if(peer==this||!peer||!peer.Body)continue;float distance=Vector3.Distance(Body.position,peer.Body.position);if(distance>.70f)continue;
                rng.NextDouble(); // Preserve the version-1 clean-scan noise stream.
                int walls=Physics.Linecast(Body.position+world.Terrain.Up*.01f,peer.Body.position+world.Terrain.Up*.01f,1<<6)?1:0;
                double rssi=-45-24*Math.Log10(Math.Max(distance,.04))-(walls*world.WallLoss)+Noise(5);
                if(rssi<-68)continue;rssi=Math.Min(-10,rssi);world.Emit(Id,"PEER_HEARD",Math.Round(rssi*2)/2,peer.Id,.85);lastHeard[peer.Id]=world.Step;
            }
            var keys=new List<string>(lastHeard.Keys);foreach(var id in keys)if(world.Step-lastHeard[id]>35){world.Emit(Id,"PEER_TIMEOUT",0,id,.6);lastHeard.Remove(id);}
        }
        public long NodeTime(long receiverMs)=>receiverMs+clockOffset+(long)(receiverMs*.0001);
        float Noise(float sigma){double u=Math.Max(1e-9,rng.NextDouble());return(float)(Math.Sqrt(-2*Math.Log(u))*Math.Cos(2*Math.PI*rng.NextDouble())*sigma);}
    }
}
