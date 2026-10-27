using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu.Rescue
{
    [Serializable]
    public sealed class SceneSpec
    {
        public string Id, Name, Disaster, Feature;
        public int Difficulty, Layout, Victims;
        public float Loss, Noise, Dust;
        public SceneSpec(string id, string name, string disaster, string feature, int difficulty, int layout, int victims)
        {
            Id = id; Name = name; Disaster = disaster; Feature = feature;
            Difficulty = difficulty; Layout = layout; Victims = victims;
            Loss = .02f + difficulty * .08f; Noise = .10f + difficulty * .13f; Dust = difficulty * .12f;
        }
    }

    /// Only this simulator/evaluator owns ground-truth geometry and source locations.
    public sealed class RescueScene : IDisposable
    {
        public const float Scale = .25f; // Reference robot 28 cm -> requested 7 cm.
        public const float Cell = .05f;
        public static readonly SceneSpec[] Catalog = {
            new SceneSpec("E01", "傾いた壁の下", "地震", "単一空隙・低雑音", 0, 0, 1),
            new SceneSpec("E02", "落下した天井材", "地震", "散在瓦礫・回り込み", 0, 1, 1),
            new SceneSpec("E03", "土砂の排水路", "斜面崩壊", "砂礫・円形暗渠", 0, 2, 1),
            new SceneSpec("E04", "浸水後の通路", "洪水", "浅い残留水・流木", 0, 3, 1),
            new SceneSpec("E05", "無人の倉庫跡", "地震", "負例・人以外の機械音", 0, 4, 0),
            new SceneSpec("M01", "パンケーキ状崩壊", "地震", "低い床板・二段空隙", 1, 5, 1),
            new SceneSpec("M02", "V字に落ちた床", "地震", "傾斜・鉄筋・分岐", 1, 6, 2),
            new SceneSpec("M03", "土砂で分断された暗渠", "斜面崩壊", "分岐・閉塞・多重反射", 1, 7, 1),
            new SceneSpec("M04", "流木の挟まった配管", "洪水", "段差・湿潤摩擦", 1, 8, 1),
            new SceneSpec("M05", "反響する無人地下室", "地震・浸水", "負例・周期的な滴水音", 1, 9, 0),
            new SceneSpec("H01", "多層床板の空隙", "地震", "三層・袋小路・欠損", 2, 10, 2),
            new SceneSpec("H02", "鉄筋が露出した階段", "地震", "屈曲・横転・不安定片", 2, 11, 1),
            new SceneSpec("H03", "岩塊で塞がれた斜面", "斜面崩壊", "急傾斜・微小な二次移動", 2, 12, 1),
            new SceneSpec("H04", "半水没した設備空間", "洪水", "反射・熱遮蔽・部分閉塞", 2, 13, 2),
            new SceneSpec("H05", "静かな深部空隙", "地震", "低発声・厚い遮蔽・負例音源", 2, 14, 1)
        };

        public readonly GameObject Root = new GameObject("RescueTruthGeometry");
        public readonly List<Vector3> VictimPositions = new List<Vector3>();
        public readonly List<AcousticSource> Sources = new List<AcousticSource>();
        public readonly List<Renderer> Roofs = new List<Renderer>();
        public readonly List<Renderer> VictimRenderers = new List<Renderer>();
        public readonly List<Rigidbody> LoosePieces = new List<Rigidbody>();
        public SceneSpec Spec;
        public Material Concrete, Stone, Rust, Ground, Water;
        public PhysicsMaterial Contact;
        readonly List<UnityEngine.Object> resources = new List<UnityEngine.Object>();
        System.Random random;

        public sealed class AcousticSource
        {
            public Vector3 Position;
            public float VoiceBand, Modulation, Power, Warmth;
            public bool Human; // Not serialized. Features are noisy observations, not this label.
        }

        public RescueScene(int index, int seed)
        {
            Spec = Catalog[Mathf.Clamp(index, 0, 14)]; random = new System.Random(seed);
            Concrete = Material(new Color(.46f, .47f, .43f), .08f);
            Stone = Material(new Color(.32f, .30f, .26f), .04f);
            Rust = Material(new Color(.36f, .19f, .10f), .28f);
            Ground = Material(new Color(.23f, .26f, .26f), .05f);
            Water = Material(new Color(.12f, .29f, .32f), .7f);
            bool wet=Spec.Disaster.Contains("洪水")||Spec.Disaster.Contains("浸水");
            Contact = new PhysicsMaterial("RubbleContact") { dynamicFriction = wet?.15f:.38f + Spec.Difficulty * .08f, staticFriction = wet?.22f:.48f, bounciness = .025f };
            resources.Add(Contact);
            Box("Foundation", new Vector3(0, -.16f, 5.1f), new Vector3(6.0f, .3f, 12.8f), Ground);
            Box("BrokenSideLeft", new Vector3(-2.8f, .6f, 5), new Vector3(.3f, 1.4f, 12), Concrete);
            Box("BrokenSideRight", new Vector3(2.8f, .6f, 5), new Vector3(.3f, 1.4f, 12), Concrete);
            int layout = Spec.Layout;
            if (layout % 5 == 0)
            {
                Roof("LeaningSlab", new Vector3(-.2f, .85f, 4.5f), new Vector3(4.7f, .24f, 6.5f), new Vector3(0, 0, 13 + Spec.Difficulty * 6));
                Box("WallPier", new Vector3(-2.0f, .4f, 4.5f), new Vector3(.45f, .9f, 6.5f), Concrete);
            }
            else if (layout % 5 == 1)
            {
                for (int i = 0; i < 4; i++) Roof("FallenFloor", new Vector3((i % 2 == 0 ? -1 : 1) * .6f, .7f + i * .12f, 2.0f + i * 2), new Vector3(4.3f, .18f, 1.7f), new Vector3(i % 2 * 9, (i - 2) * 7, i % 2 == 0 ? 11 : -11));
            }
            else if (layout % 5 == 2)
            {
                for (int i = 0; i < 8; i++)
                {
                    float z = 1 + i * 1.25f;
                    Box("CulvertLeft", new Vector3(-1.15f, .35f, z), new Vector3(.3f, .8f, 1.15f), Stone);
                    Box("CulvertRight", new Vector3(1.15f, .35f, z), new Vector3(.3f, .8f, 1.15f), Stone);
                    Roof("CulvertRoof", new Vector3(0, .82f, z), new Vector3(2.6f, .18f, 1.15f), new Vector3(0, 0, (i % 3 - 1) * 4));
                }
            }
            else if (layout % 5 == 3)
            {
                for (int i = 0; i < 6; i++)
                {
                    var log = Box("Driftwood", new Vector3((i % 2 == 0 ? -.65f : .65f), .13f, 1.8f + i * 1.55f), new Vector3(2.1f, .18f, .24f), Stone);
                    log.transform.localRotation = Quaternion.Euler(0, (i % 2 == 0 ? 32 : -32), 6);
                }
                var water = Box("ShallowResidualWaterVisual", new Vector3(0, .015f, 7), new Vector3(4.8f, .008f, 6), Water);
                UnityEngine.Object.DestroyImmediate(water.GetComponent<Collider>());
            }
            else
            {
                Box("Partition", new Vector3(.45f, .36f, 5), new Vector3(.24f, .8f, 6.5f), Concrete);
                Roof("BasementRoof", new Vector3(0, 1.2f, 6), new Vector3(5.2f, .22f, 8), new Vector3(4, 0, 0));
            }
            if (Spec.Difficulty > 0)
            {
                Roof("CollapsedLowerDeck", new Vector3(-.9f, .52f, 6.3f), new Vector3(2.3f, .16f, 4.3f), new Vector3(5, 0, -8));
                Box("NarrowThroatLeft", new Vector3(-.65f, .19f, 3.2f), new Vector3(.65f, .55f, .7f), Concrete);
                Box("NarrowThroatRight", new Vector3(.65f, .19f, 3.2f), new Vector3(.65f, .55f, .7f), Concrete);
            }
            if (Spec.Difficulty == 2)
            {
                Roof("UpperBrokenDeck", new Vector3(.7f, 1.55f, 8), new Vector3(3.2f, .24f, 4.2f), new Vector3(-9, 4, 13));
                Box("BlindAlley", new Vector3(-1.6f, .25f, 8.8f), new Vector3(1.7f, .65f, .28f), Concrete);
                Box("RaisedStep", new Vector3(0, .055f, 4.8f), new Vector3(1.5f, .11f, .5f), Stone);
            }
            int rubbleCount = 28 + Spec.Difficulty * 22;
            for (int i = 0; i < rubbleCount; i++)
            {
                float x = Range(-2.5f, 2.5f), z = Range(1.4f, 11.1f);
                // A traversable but uneven path, not a hidden precomputed robot route.
                float lane = Mathf.Sin(z * .65f + layout) * .55f;
                if (Mathf.Abs(x - lane) < .34f) x += x > lane ? .65f : -.65f;
                var size = new Vector3(Range(.14f,.55f), Range(.08f,.32f), Range(.16f,.65f));
                var g = Box("FracturedConcrete", new Vector3(x, size.y * .3f, z), size, i % 3 == 0 ? Stone : Concrete);
                Fracture(g);
                g.transform.localRotation = Quaternion.Euler(Range(-14,14),Range(0,180),Range(-18,18));
                if (Spec.Difficulty == 2 && i % 17 == 0)
                {
                    var body = g.AddComponent<Rigidbody>(); body.mass = .08f; body.linearDamping = 2; body.angularDamping = 2; LoosePieces.Add(body);
                }
            }
            for (int i = 0; i < 9 + Spec.Difficulty * 8; i++)
            {
                var bar = GameObject.CreatePrimitive(PrimitiveType.Cylinder); bar.name = "ExposedRebar"; bar.layer = 6; bar.transform.SetParent(Root.transform,false);
                bar.transform.localPosition = new Vector3(Range(-2.1f,2.1f)*Scale, .2f*Scale, Range(2,11)*Scale);
                bar.transform.localScale = new Vector3(.025f,Range(.18f,.5f),.025f)*Scale;
                bar.transform.localRotation = Quaternion.Euler(70,Range(0,180),Range(-25,25)); bar.GetComponent<Renderer>().sharedMaterial = Rust;
            }
            var sourceRandom=new System.Random(seed ^ 0x4f1bbcdc ^ (index*397));
            for (int i = 0; i < Spec.Victims; i++)
            {
                Physics.SyncTransforms();var p=SampleSourcePocket(sourceRandom);
                VictimPositions.Add(p);
                Sources.Add(new AcousticSource { Position=p, Human=true, VoiceBand=.84f, Modulation=layout==14?.32f:.84f, Power=layout==14?.32f:1, Warmth=3.6f });
                var marker = GameObject.CreatePrimitive(PrimitiveType.Capsule); marker.name="EvaluationOnlyPerson";marker.transform.SetParent(Root.transform,false);marker.transform.position=p;marker.transform.localScale=new Vector3(.13f,.22f,.13f)*Scale;marker.transform.rotation=Quaternion.Euler(75,0,10);UnityEngine.Object.DestroyImmediate(marker.GetComponent<Collider>());var renderer=marker.GetComponent<Renderer>();renderer.sharedMaterial=Material(new Color(.95f,.52f,.2f),.1f);renderer.enabled=false;VictimRenderers.Add(renderer);
            }
            Sources.Add(new AcousticSource { Position=new Vector3(1.7f,.25f,6.0f)*Scale, VoiceBand=layout==9?.73f:.25f, Modulation=layout==9?.72f:.28f, Power=.8f, Warmth=layout==4?1.5f:0 });
            Physics.SyncTransforms(); Cutaway(true);
        }

        float Range(float a,float b) => a+(b-a)*(float)random.NextDouble();
        Vector3 SampleSourcePocket(System.Random sourceRandom)
        {
            // Separate RNG: changing the acoustic placement never changes rubble geometry.
            // Sources represent the mouth/heat observation point, not a scaled human body.
            for(int attempt=0;attempt<512;attempt++)
            {
                float x=-.50f+(float)sourceRandom.NextDouble();
                float y=.04f+(float)sourceRandom.NextDouble()*.08f;
                float z=Spec.Layout==14?1.70f+(float)sourceRandom.NextDouble()*.95f:.55f+(float)sourceRandom.NextDouble()*2.0f;
                var p=new Vector3(x,y,z);if(Physics.CheckSphere(p,.012f,1<<6))continue;
                bool separate=true;foreach(var previous in VictimPositions)if(Vector3.Distance(previous,p)<.55f){separate=false;break;}
                if(separate)return p;
            }
            throw new InvalidOperationException("Could not sample independent source pockets for "+Spec.Id);
        }
        Vector3 FindAirPocket(Vector3 preferred)
        {
            if(!Physics.CheckSphere(preferred,.012f,1<<6))return preferred;
            Vector3 best=preferred;float distance=float.PositiveInfinity;
            for(int radius=1;radius<=12;radius++)
            {
                for(int x=-radius;x<=radius;x++)for(int y=-radius;y<=radius;y++)for(int z=-radius;z<=radius;z++)
                {
                    if(Mathf.Max(Mathf.Abs(x),Mathf.Abs(y),Mathf.Abs(z))!=radius)continue;
                    var candidate=preferred+new Vector3(x,y,z)*.0125f;if(candidate.y<.025f||candidate.y>.35f)continue;
                    float d=(candidate-preferred).sqrMagnitude;if(d>=distance||Physics.CheckSphere(candidate,.012f,1<<6))continue;best=candidate;distance=d;
                }
                if(float.IsFinite(distance))return best;
            }
            throw new InvalidOperationException("No physical survival pocket near source in "+Spec.Id);
        }
        void Fracture(GameObject g)
        {
            var mesh=UnityEngine.Object.Instantiate(g.GetComponent<MeshFilter>().sharedMesh);mesh.name="SeededBrokenConcrete";resources.Add(mesh);
            var vertices=mesh.vertices;var corners=new Dictionary<Vector3,Vector3>();
            for(int i=0;i<vertices.Length;i++){var p=vertices[i];if(!corners.TryGetValue(p,out Vector3 shifted)){shifted=p+new Vector3(Range(-.13f,.13f),Range(-.13f,.13f),Range(-.13f,.13f));corners.Add(p,shifted);}vertices[i]=shifted;}
            mesh.vertices=vertices;mesh.RecalculateNormals();mesh.RecalculateBounds();g.GetComponent<MeshFilter>().sharedMesh=mesh;UnityEngine.Object.DestroyImmediate(g.GetComponent<Collider>());var collider=g.AddComponent<MeshCollider>();collider.sharedMesh=mesh;collider.convex=true;collider.sharedMaterial=Contact;
        }
        Material Material(Color color,float gloss)
        {
            var material = new Material(Shader.Find("Standard")); material.color=color;material.SetFloat("_Glossiness",gloss);resources.Add(material);return material;
        }
        GameObject Box(string name,Vector3 p,Vector3 size,Material material)
        {
            var g=GameObject.CreatePrimitive(PrimitiveType.Cube);g.name=name;g.layer=6;g.transform.SetParent(Root.transform,false);g.transform.localPosition=p*Scale;g.transform.localScale=size*Scale;g.GetComponent<Renderer>().sharedMaterial=material;g.GetComponent<Collider>().sharedMaterial=Contact;return g;
        }
        void Roof(string name,Vector3 p,Vector3 size,Vector3 rotation)
        {
            var g=Box(name,p,size,Concrete);g.transform.localRotation=Quaternion.Euler(rotation);Roofs.Add(g.GetComponent<Renderer>());
        }
        public void Cutaway(bool enabled) {foreach(var r in Roofs)r.enabled=!enabled;}
        public void ShowTruth(bool enabled) {foreach(var r in VictimRenderers)r.enabled=enabled;}
        public void Aftershock(int step)
        {
            if(Spec.Difficulty<2 || step!=1000)return;
            foreach(var body in LoosePieces)if(body)body.AddForce(new Vector3(.015f,.006f,-.01f),ForceMode.Impulse);
        }
        public void Dispose()
        {
            if(Root)UnityEngine.Object.DestroyImmediate(Root);
            foreach(var resource in resources)if(resource)UnityEngine.Object.DestroyImmediate(resource);
        }
    }
}
