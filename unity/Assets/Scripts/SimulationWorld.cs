using System;
using System.Collections.Generic;
using UnityEngine;
namespace Purumogu {
    public sealed class SimulationWorld:IDisposable {
        public const float Dt=.02f;
        public readonly ScenarioBuilder Terrain=new ScenarioBuilder();
        public readonly List<NodeBody> Nodes=new List<NodeBody>();
        public readonly List<SensorEvent> Pending=new List<SensorEvent>();
        public GateActuator[] Gates;
        public Material NodeMaterial,ArmMaterial,LinkMaterial,GateMaterial;
        public PhysicsMaterial NodePhysics;
        public int Step,SensorSeed,PhysicsSeed,Deployed;
        public double PacketLoss;public float WallLoss=28;
        public string Mode="CLUSTER";public bool Passive;
        public float MaxDepth;public int PassedNodes;
        readonly HashSet<string> passed=new HashSet<string>();
        public void Build(int family,int mapSeed,int physicsSeed,int sensorSeed,double loss=0,bool materialOOD=false){
            Physics.simulationMode=SimulationMode.Script;Physics.gravity=new Vector3(0,-9.81f,0);Physics.defaultSolverIterations=10;Physics.defaultSolverVelocityIterations=4;
            PhysicsSeed=physicsSeed;SensorSeed=sensorSeed;PacketLoss=loss;Terrain.Build(family,mapSeed,materialOOD);
            NodeMaterial=ScenarioBuilder.Mat(new Color(.85f,.90f,.83f));ArmMaterial=ScenarioBuilder.Mat(new Color(.12f,.78f,.66f));GateMaterial=ScenarioBuilder.Mat(new Color(.95f,.57f,.22f));LinkMaterial=new Material(Shader.Find("Sprites/Default")){color=new Color(.25f,1f,.80f)};
            NodePhysics=new PhysicsMaterial("NodeContact"){dynamicFriction=.10f,staticFriction=.13f,bounciness=.025f,frictionCombine=PhysicsMaterialCombine.Average};
            Gates=new[]{new GateActuator(this,0),new GateActuator(this,1),new GateActuator(this,2)};
        }
        public bool Deploy(ActionCommand a,out string reason){
            if(a.entry_id<0||a.entry_id>2||a.node_count<1||a.node_count>10||Deployed+a.node_count>20||(a.formation_mode!="CLUSTER"&&a.formation_mode!="CHAIN_INSERT")||a.delay_steps!=0){reason="invalid or infeasible actuator command";return false;}
            Mode=a.formation_mode;Passive=a.policy=="passive";Emit("PARENT","PARENT_CMD",Mode=="CHAIN_INSERT"?1:0);
            var rng=new System.Random(PhysicsSeed+Deployed*7919);
            for(int i=0;i<a.node_count;i++){Vector3 p=Terrain.Entry(a.entry_id)+Terrain.Forward*((i-(a.node_count-1)*.5f)*.052f)+Terrain.Up*(.005f+(float)rng.NextDouble()*.003f);string id="N"+Deployed.ToString("D2");var node=NodeBody.Create(this,id,p,SensorSeed+Deployed*101);Nodes.Add(node);Deployed++;}
            foreach(var node in Nodes)node.Command(Mode,!Passive);Gates[a.entry_id].Arm();Physics.SyncTransforms();reason="";return true;
        }
        public void Tick(){foreach(var gate in Gates)gate.Tick();foreach(var node in Nodes)if(node)node.MotorStep();Physics.Simulate(Dt);Step++;foreach(var node in Nodes){if(!node)continue;node.Sample();float depth=Terrain.LocalDepth(node.Body.position);MaxDepth=Mathf.Max(MaxDepth,depth);if(depth>11.4f*Terrain.Cell)passed.Add(node.Id);}PassedNodes=passed.Count;}
        public void Emit(string id,string kind,double value=0,string peer="",double quality=1){if(Pending.Count>=12000)throw new InvalidOperationException("sensor event buffer exceeded");long ms=Step*20L;long nodeMs=ms;foreach(var n in Nodes)if(n.Id==id){nodeMs=n.NodeTime(ms);break;}Pending.Add(new SensorEvent{node_id=id,kind=kind,t_ms=ms,node_time_ms=nodeMs,peer_id=peer,value=value,quality=quality});}
        public void SafeStop(){foreach(var gate in Gates)gate.Stop();foreach(var node in Nodes)node.Command("CLUSTER",false);}
        public void Dispose(){foreach(var n in Nodes)if(n)UnityEngine.Object.DestroyImmediate(n.gameObject);Nodes.Clear();Terrain.Dispose();if(NodePhysics)UnityEngine.Object.DestroyImmediate(NodePhysics);foreach(var m in new[]{NodeMaterial,ArmMaterial,LinkMaterial,GateMaterial})if(m)UnityEngine.Object.DestroyImmediate(m);Pending.Clear();}
    }
}
