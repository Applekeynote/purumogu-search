using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu.Rescue
{
    /// Three actual rigid bodies, not animated transforms or a single box collider.
    public sealed class RescueRobot : IDisposable
    {
        public const float Length = .07f;
        public const float MaxRange = 2.0f;
        public string Id;
        public GameObject Root;
        public Rigidbody Core;
        public readonly Rigidbody[] Cones = new Rigidbody[2];
        public readonly ConfigurableJoint[] Hinges = new ConfigurableJoint[2];
        public readonly bool[] PortBusy = new bool[2];
        public Vector3 DeploymentAnchor, EstimatedPosition;
        public float Energy=1, Impact, LastSnr, PoseSigma, Temperature, Signal=-100;
        public int Peers, Received, Dropped;
        /// Offline training label only; this robot object is never serialized to Rust.
        public bool EvaluationSourceIsHuman { get; private set; }
        public bool Powered=true, Stuck;
        public bool Lost { get; private set; }
        public bool Released { get; private set; }=true;
        public int ReleaseStep { get; private set; }
        public int HoldUntilStep;
        public bool Verifying { get; private set; }
        public Vector3 ExplorationTarget;
        public bool HasTarget;
        public float BendDegrees, SpinDegrees;
        readonly List<UnityEngine.Object> resources=new List<UnityEngine.Object>();
        readonly System.Random random;
        Vector3 previousSensorPosition, pendingOdometry;
        int progressSteps,avoidUntil;
        Vector3 progressPosition,avoidHeading;
        float yawBias;
        float rotorPhase;

        public RescueRobot(string id,Vector3 position,int seed)
        {
            Id=id;random=new System.Random(seed);DeploymentAnchor=EstimatedPosition=previousSensorPosition=position;
            progressPosition=position;
            Root=new GameObject(id+"_ArticulatedRobot");
            var shell=MakeMaterial(new Color(.56f,.59f,.55f),.78f,.7f);
            var dark=MakeMaterial(new Color(.055f,.075f,.075f),.9f,.62f);
            var trim=MakeMaterial(new Color(.14f,.85f,.72f),.35f,.7f);
            var contact=new PhysicsMaterial("RobotGrip"){dynamicFriction=.17f,staticFriction=.28f,bounciness=.01f};resources.Add(contact);
            var ball=new GameObject("CentralSphericalJoint");ball.transform.SetParent(Root.transform,false);ball.transform.position=position;
            var sphere=GameObject.CreatePrimitive(PrimitiveType.Sphere);sphere.transform.SetParent(ball.transform,false);sphere.transform.localScale=Vector3.one*.019f;sphere.GetComponent<Renderer>().sharedMaterial=dark;UnityEngine.Object.DestroyImmediate(sphere.GetComponent<Collider>());
            Core=Body(ball,.008f);var ballCollider=ball.AddComponent<SphereCollider>();ballCollider.radius=.0095f;ballCollider.sharedMaterial=contact;
            for(int side=0;side<2;side++)
            {
                int sign=side==0?1:-1;
                var g=new GameObject(side==0?"FrontTaperedCone":"RearTaperedCone");g.transform.SetParent(Root.transform,false);g.transform.position=position+Vector3.forward*(sign*.0075f);
                var mesh=ConeMesh(sign);resources.Add(mesh);g.AddComponent<MeshFilter>().sharedMesh=mesh;g.AddComponent<MeshRenderer>().sharedMaterial=shell;
                var collider=g.AddComponent<MeshCollider>();collider.sharedMesh=mesh;collider.convex=true;collider.sharedMaterial=contact;Cones[side]=Body(g,.008f);
                var joint=g.AddComponent<ConfigurableJoint>();joint.connectedBody=Core;joint.autoConfigureConnectedAnchor=false;joint.anchor=Vector3.zero;joint.connectedAnchor=Vector3.forward*(sign*.0075f);
                joint.xMotion=joint.yMotion=joint.zMotion=ConfigurableJointMotion.Locked;
                joint.axis=Vector3.forward;joint.secondaryAxis=Vector3.up;
                joint.angularXMotion=ConfigurableJointMotion.Free;joint.angularYMotion=joint.angularZMotion=ConfigurableJointMotion.Limited;
                joint.angularYLimit=joint.angularZLimit=new SoftJointLimit{limit=65};
                joint.rotationDriveMode=RotationDriveMode.Slerp;
                joint.slerpDrive=new JointDrive{positionSpring=.010f,positionDamper=.00012f,maximumForce=.005f};
                joint.enableCollision=false;joint.projectionMode=JointProjectionMode.PositionAndRotation;joint.projectionDistance=.001f;joint.projectionAngle=8;Hinges[side]=joint;
                // Spherical seating rings and the dark flattened inner panel of the reference.
                var ring=GameObject.CreatePrimitive(PrimitiveType.Sphere);ring.name="BallSeat";ring.transform.SetParent(g.transform,false);ring.transform.localScale=new Vector3(.018f,.018f,.009f);ring.GetComponent<Renderer>().sharedMaterial=shell;UnityEngine.Object.DestroyImmediate(ring.GetComponent<Collider>());
                var panel=GameObject.CreatePrimitive(PrimitiveType.Cube);panel.name="ConeSensorPanel";panel.transform.SetParent(g.transform,false);panel.transform.localPosition=new Vector3(0,.0067f,sign*.0115f);panel.transform.localScale=new Vector3(.0095f,.0005f,.014f);panel.transform.localRotation=Quaternion.Euler(sign*12,0,0);panel.GetComponent<Renderer>().sharedMaterial=dark;UnityEngine.Object.DestroyImmediate(panel.GetComponent<Collider>());
                var tip=GameObject.CreatePrimitive(PrimitiveType.Sphere);tip.name="MagneticDockPort";tip.transform.SetParent(g.transform,false);tip.transform.localPosition=Vector3.forward*(sign*.0265f);tip.transform.localScale=new Vector3(.0045f,.0045f,.002f);tip.GetComponent<Renderer>().sharedMaterial=trim;UnityEngine.Object.DestroyImmediate(tip.GetComponent<Collider>());
            }
            foreach(var a in Core.GetComponents<Collider>())foreach(var cone in Cones)foreach(var b in cone.GetComponents<Collider>())Physics.IgnoreCollision(a,b);
        }
        Material MakeMaterial(Color color,float metal,float gloss)
        {
            var m=new Material(Shader.Find("Standard"));m.color=color;m.SetFloat("_Metallic",metal);m.SetFloat("_Glossiness",gloss);resources.Add(m);return m;
        }
        static Rigidbody Body(GameObject g,float mass)
        {
            var body=g.AddComponent<Rigidbody>();body.mass=mass;body.linearDamping=3.5f;body.angularDamping=3;body.maxAngularVelocity=18;body.solverIterations=16;body.solverVelocityIterations=8;body.collisionDetectionMode=CollisionDetectionMode.ContinuousDynamic;return body;
        }
        static Mesh ConeMesh(int sign)
        {
            const int sides=24;float[] z={0,.005f,.021f,.0265f};float[] radius={.0088f,.0092f,.0037f,.0016f};
            var vertices=new List<Vector3>();var triangles=new List<int>();
            for(int ring=0;ring<z.Length;ring++)for(int i=0;i<sides;i++)
            {
                float a=i*Mathf.PI*2/sides;float y=Mathf.Sin(a)*radius[ring];y=Mathf.Clamp(y,-radius[ring]*.74f,radius[ring]*.74f);
                vertices.Add(new Vector3(Mathf.Cos(a)*radius[ring],y,sign*z[ring]));
            }
            for(int ring=0;ring<3;ring++)for(int i=0;i<sides;i++)
            {
                int a=ring*sides+i,b=ring*sides+(i+1)%sides,c=a+sides,d=b+sides;
                if(sign>0)triangles.AddRange(new[]{a,b,c,b,d,c});else triangles.AddRange(new[]{a,c,b,b,c,d});
            }
            int first=vertices.Count;vertices.Add(Vector3.zero);vertices.Add(Vector3.forward*(sign*z[3]));
            for(int i=0;i<sides;i++){int next=(i+1)%sides;if(sign>0)triangles.AddRange(new[]{first,next,i,first+1,3*sides+i,3*sides+next});else triangles.AddRange(new[]{first,i,next,first+1,3*sides+next,3*sides+i});}
            var mesh=new Mesh{name="FlattenedRoundedTaper"};mesh.SetVertices(vertices);mesh.SetTriangles(triangles,0);mesh.RecalculateNormals();mesh.RecalculateBounds();return mesh;
        }
        public Vector3 Port(int side)=>Cones[side].transform.TransformPoint(Vector3.forward*(side==0?.0265f:-.0265f));
        public Vector3 PortDirection(int side)=>Cones[side].transform.forward*(side==0?1:-1);
        public void ConfigureDrop(Quaternion rotation,int releaseStep)
        {
            foreach(var body in new[]{Core,Cones[0],Cones[1]}){body.position=DeploymentAnchor+rotation*(body.position-DeploymentAnchor);body.rotation=rotation*body.rotation;body.isKinematic=true;}
            ReleaseStep=releaseStep;Released=false;
        }
        public void ReleaseAt(int step)
        {
            if(Released||step<ReleaseStep)return;Released=true;
            foreach(var body in new[]{Core,Cones[0],Cones[1]}){body.isKinematic=false;body.WakeUp();}
        }
        public void Motor(int step)
        {
            Verifying=step<HoldUntilStep;
            if(!Released||!Powered||Energy<=0)return;
            if(Verifying){Energy=Mathf.Max(0,Energy-.000001f);return;}
            Vector3 heading=Vector3.forward;
            if(HasTarget){heading=Vector3.ProjectOnPlane(ExplorationTarget-EstimatedPosition,Vector3.up);if(heading.magnitude>.018f)heading.Normalize();else heading=Vector3.forward;}
            if(EstimatedPosition.z>2.65f)heading=Vector3.back;
            if(Mathf.Abs(EstimatedPosition.x)>.61f)heading=new Vector3(-Mathf.Sign(EstimatedPosition.x),0,0);
            Vector3 nose=Core.position+Vector3.up*.005f;
            bool lowBlocked=Physics.SphereCast(nose,.007f,heading,out RaycastHit hit,.10f,1<<6)&&hit.normal.y<.6f;
            bool stepUp=lowBlocked&&!Stuck&&!Physics.SphereCast(nose+Vector3.up*.033f,.007f,heading,out _,.10f,1<<6);
            if(lowBlocked&&!stepUp)
            {
                Vector3 desired=heading,best=heading;float bestScore=float.NegativeInfinity;
                foreach(float yaw in new[]{0f,-25,25,-50,50,-80,80,-115,115,180})
                {
                    Vector3 candidate=Quaternion.Euler(0,yaw,0)*desired;float clearance=Physics.SphereCast(nose,.008f,candidate,out RaycastHit obstacle,.14f,1<<6)?obstacle.distance:.14f;
                    float score=clearance/.14f*1.4f+Vector3.Dot(candidate,desired)*.65f;if(score>bestScore){bestScore=score;best=candidate;}
                }
                heading=best;avoidHeading=best;avoidUntil=step+150;
            }
            if(step<avoidUntil)heading=avoidHeading;
            if(Stuck&&step%180<70)heading=(Id[Id.Length-1]%2==0?Vector3.left:Vector3.right)*.8f-Vector3.forward*.4f;
            if(EstimatedPosition.z<-.20f)heading=Vector3.forward;
            if(EstimatedPosition.z>2.65f)heading=Vector3.back;
            if(Mathf.Abs(EstimatedPosition.x)>.61f)heading=new Vector3(-Mathf.Sign(EstimatedPosition.x),0,0);
            heading.Normalize();float speed=Vector3.Dot(Core.linearVelocity,heading);
            int grounded=Physics.Raycast(Core.position,Vector3.down,.032f,1<<6)?1:0;foreach(var cone in Cones)if(Physics.Raycast(cone.worldCenterOfMass,Vector3.down,.032f,1<<6))grounded++;
            float force=.14f*Mathf.Clamp01(1-speed/.12f)/Mathf.Max(grounded,1);
            if(Physics.Raycast(Core.position,Vector3.down,.032f,1<<6))Core.AddForce(heading*force,ForceMode.Force);foreach(var cone in Cones)if(Physics.Raycast(cone.worldCenterOfMass,Vector3.down,.032f,1<<6))cone.AddForce(heading*force,ForceMode.Force);
            float phase=step*.02f*5+int.Parse(Id.Substring(1))*.45f;
            if(!stepUp)rotorPhase=(rotorPhase+1.5f)%360;
            for(int i=0;i<2;i++)
            {
                var desired=Quaternion.Euler(Mathf.Sin(phase+i*Mathf.PI)*20,Mathf.Cos(phase*.7f+i)*24,i==0?rotorPhase:-rotorPhase);
                if(stepUp)desired=Quaternion.Euler(i==0?-55:-35,0,i==0?rotorPhase:-rotorPhase);
                var jointSpace=Quaternion.LookRotation(Vector3.left,Vector3.up);
                Hinges[i].targetRotation=Quaternion.Inverse(jointSpace)*Quaternion.Inverse(desired)*jointSpace;
                if(!stepUp)Cones[i].AddRelativeTorque(Vector3.forward*(i==0?.000004f:-.000004f));
            }
            Energy=Mathf.Max(0,Energy-.000006f);
        }
        public void AfterPhysics()
        {
            if(!Released)return;
            if(!Lost&&(!float.IsFinite(Core.position.x)||!float.IsFinite(Core.position.y)||!float.IsFinite(Core.position.z)||Core.position.y<-.12f||Mathf.Abs(Core.position.x)>1.0f||Core.position.z<-.6f||Core.position.z>3.6f))
            {
                Lost=true;Powered=false;
                foreach(var body in new[]{Core,Cones[0],Cones[1]}){body.linearVelocity=Vector3.zero;body.angularVelocity=Vector3.zero;body.isKinematic=true;}
            }
            if(++progressSteps>=100){Stuck=Vector3.Distance(progressPosition,EstimatedPosition)<.03f;progressPosition=EstimatedPosition;progressSteps=0;}
            BendDegrees=Vector3.Angle(Core.transform.forward,Cones[0].transform.forward);SpinDegrees=(Quaternion.Inverse(Core.rotation)*Cones[0].rotation).eulerAngles.z;
        }
        public SensorScan Observe(RescueScene scene,int step,IReadOnlyList<RescueRobot> peers)
        {
            if(!Released||Lost)return null;
            var actualDelta=Core.position-previousSensorPosition;previousSensorPosition=Core.position;
            yawBias+=Noise(.018f);var drift=Quaternion.Euler(Noise(.6f),yawBias+Noise(.6f),Noise(.6f));
            var delta=drift*actualDelta*(1+Noise(.025f))+new Vector3(Noise(.00008f),Noise(.00008f),Noise(.00008f));EstimatedPosition+=delta;pendingOdometry+=delta;
            float angularSigma=.6f*Mathf.Deg2Rad;
            PoseSigma=Mathf.Sqrt(PoseSigma*PoseSigma+.000025f*actualDelta.magnitude+(.025f*.025f+2*angularSigma*angularSigma)*actualDelta.sqrMagnitude+3*.00008f*.00008f);
            var scan=new SensorScan{node_id=Id,deployment_anchor=Array(DeploymentAnchor),odometry_delta=Array(pendingOdometry),pose_sigma_m=PoseSigma};
            for(int ring=0;ring<4;ring++)for(int i=0;i<(ring==0?32:ring==1?16:8);i++)
            {
                int count=ring==0?32:ring==1?16:8;float yaw=i*360f/count+(step/25%4)*2.8125f,pitch=ring==0?0:ring==1?-15:ring==2?-35:30;var local=Quaternion.Euler(pitch,yaw,0)*Vector3.forward;
                var direction=Core.rotation*local;bool hit=Physics.Raycast(Core.position,direction,out RaycastHit result,MaxRange,1<<6);
                float range=hit?Mathf.Clamp(result.distance+Noise(.002f+result.distance*.003f+scene.Spec.Dust*.006f),.002f,MaxRange):MaxRange;
                if(random.NextDouble()<scene.Spec.Dust*.15f)continue;
                scan.rays.Add(new RangeRay{direction=Array((drift*direction).normalized),distance_m=range,hit=hit});
            }
            RescueScene.AcousticSource strongest=null;float strongestPower=0,ambientPower=.018f+scene.Spec.Noise*.12f,totalSourcePower=0,voicePower=ambientPower*.35f,modulationPower=ambientPower*.35f;Vector3 bearing=Vector3.zero;
            foreach(var source in scene.Sources)
            {
                float distance=Vector3.Distance(Core.position,source.Position);
                bool occluded=Physics.Linecast(Core.position,source.Position,1<<6);
                float power=source.Power/(1+distance*distance*12)*(occluded?.18f:1);
                power*=.75f+.25f*Mathf.Sin(step*.02f*3.5f+source.Position.z*9);
                if(power>strongestPower){strongest=source;strongestPower=power;bearing=(source.Position-Core.position).normalized;}
                totalSourcePower+=power;voicePower+=power*source.VoiceBand;modulationPower+=power*source.Modulation;
            }
            float noiseAndInterference=ambientPower+Mathf.Max(0,totalSourcePower-strongestPower);
            // Powers add incoherently. Use 10 log10(Psignal/Pnoise), never 20 log10.
            LastSnr=PowerRatioDb(strongestPower,noiseAndInterference)+Noise(2);
            EvaluationSourceIsHuman=strongest!=null&&strongest.Human;
            float sigma=scene.Spec.Difficulty*9+8+Mathf.Max(0,8-LastSnr);
            bool reflected=strongest!=null&&Physics.Linecast(Core.position,strongest.Position,1<<6);
            if(reflected){sigma+=18;bearing=Quaternion.Euler(0,(Id[Id.Length-1]%2==0?1:-1)*24,0)*bearing;}
            scan.bearing=Array((Quaternion.Euler(Noise(sigma*.4f),Noise(sigma*.6f),Noise(sigma*.3f))*bearing).normalized);
            scan.bearing_sigma_deg=Mathf.Clamp(sigma,3,110);scan.acoustic_snr_db=Mathf.Clamp(LastSnr,-60,80);
            scan.voice_band_ratio=Mathf.Clamp01(voicePower/(ambientPower+totalSourcePower)+Noise(.12f+scene.Spec.Noise*.1f));scan.modulation=Mathf.Clamp01(modulationPower/(ambientPower+totalSourcePower)+Noise(.15f));
            float heat=0;foreach(var source in scene.Sources){float d=Vector3.Distance(Core.position,source.Position);heat+=source.Warmth*Mathf.Exp(-d*7)*(Physics.Linecast(Core.position,source.Position,1<<6)?.1f:1);}
            Temperature=heat+Noise(.12f+scene.Spec.Difficulty*.06f);scan.temperature_delta_c=Mathf.Clamp(Temperature,-20,40);
            Peers=0;Signal=-100;foreach(var peer in peers)if(peer!=this&&peer.Released&&!peer.Lost){float distance=Vector3.Distance(Core.position,peer.Core.position);float rssi=Mathf.Clamp(-45-26*Mathf.Log10(Mathf.Max(distance,.04f))-(Physics.Linecast(Core.position,peer.Core.position,1<<6)?28:0)+Noise(4),-120,-10);if(rssi>-78){Peers++;Signal=Mathf.Max(Signal,rssi);}}
            if(random.NextDouble()<scene.Spec.Loss){Dropped++;return null;}Received++;pendingOdometry=Vector3.zero;return scan;
        }
        public static float PowerRatioDb(float signalPower,float noisePower)=>10*Mathf.Log10(Mathf.Max(signalPower,.00001f)/Mathf.Max(noisePower,.00001f));
        float Noise(float sigma){double u=Math.Max(1e-9,random.NextDouble());return(float)(Math.Sqrt(-2*Math.Log(u))*Math.Cos(2*Math.PI*random.NextDouble())*sigma);}
        public static double[] Array(Vector3 v)=>new[]{(double)v.x,v.y,v.z};
        public void Dispose(){if(Root)UnityEngine.Object.DestroyImmediate(Root);foreach(var r in resources)if(r)UnityEngine.Object.DestroyImmediate(r);}
    }

    [Serializable] public sealed class RangeRay {public double[] direction;public double distance_m;public bool hit;}
    [Serializable] public sealed class SensorScan
    {
        public string node_id;public double[] deployment_anchor,odometry_delta,bearing;public double pose_sigma_m,bearing_sigma_deg,acoustic_snr_db,voice_band_ratio,modulation,temperature_delta_c;public List<RangeRay> rays=new List<RangeRay>();
    }
    [Serializable] public sealed class RescuePacket
    {
        public string protocol_version="2.0",run_id;public long seq_no,sim_step;public double cell_m=RescueScene.Cell;public bool request_map=true,request_diagnostics=false;public List<SensorScan> scans=new List<SensorScan>();
    }
}
