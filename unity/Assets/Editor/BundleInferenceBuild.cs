using System.IO;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Purumogu.Editor
{
    public sealed class BundleInferenceBuild : IPreprocessBuildWithReport
    {
        static string Arg(string key,string fallback){var args=System.Environment.GetCommandLineArgs();for(int i=0;i<args.Length-1;i++)if(args[i]==key)return args[i+1];return fallback;}
        static void Copy(string source,string destination)
        {if(!Path.GetFullPath(source).Equals(Path.GetFullPath(destination),System.StringComparison.OrdinalIgnoreCase))File.Copy(source,destination,true);}
        public int callbackOrder=>0;
        public void OnPreprocessBuild(BuildReport report)
        {
            string root=Path.GetFullPath(Path.Combine(Application.dataPath,"../.."));
            string prepared=Path.Combine(Application.streamingAssetsPath,"inference");
            // Keep the distributed, verified runtime when rebuilding an existing app.
            // A developer can explicitly select a newly compiled server with -purumoguServer.
            string server=Path.GetFullPath(Arg("-purumoguServer",File.Exists(Path.Combine(prepared,"rescue_server.exe"))?Path.Combine(prepared,"rescue_server.exe"):Path.Combine(root,"rust/target/release/rescue_server.exe")));
            string model=Path.GetFullPath(Arg("-purumoguModel",Path.Combine(root,"models/swarm-v35-generic/model.json")));
            // A distributed Unity project already has the verified CPU bundle.
            // Explicit command-line overrides still fail if their files are missing.
            if(!File.Exists(server)&&Arg("-purumoguServer",null)==null)server=Path.Combine(prepared,"rescue_server.exe");
            if(!File.Exists(model)&&Arg("-purumoguModel",null)==null)model=Path.Combine(prepared,"model.json");
            if(!File.Exists(server)||!File.Exists(model))throw new BuildFailedException("Build rescue_server --release first; bundled model must exist.");
            string destination=Path.Combine(Application.streamingAssetsPath,"inference");Directory.CreateDirectory(destination);
            Copy(server,Path.Combine(destination,"rescue_server.exe"));
            Copy(model,Path.Combine(destination,"model.json"));
            var sceneIds=new Newtonsoft.Json.Linq.JArray();
            foreach(var scene in Purumogu.Rescue.RescueScene.Catalog)sceneIds.Add(scene.Id);
            var buildInfo=new Newtonsoft.Json.Linq.JObject{
                ["scene_count"]=sceneIds.Count,["scene_ids"]=sceneIds,["unity_version"]=Application.unityVersion
            };
            File.WriteAllText(Path.Combine(destination,"build-info.json"),buildInfo.ToString());
            var descriptor=Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(model));
            if((string)descriptor["volume_file"]=="volume.onnx")
            {
                string source=Path.GetDirectoryName(model);
                foreach(string name in new[]{"volume.onnx","onnxruntime.dll","onnxruntime_providers_shared.dll","ONNXRUNTIME-LICENSE.txt","ONNXRUNTIME-ThirdPartyNotices.txt"})
                {if(!File.Exists(Path.Combine(source,name)))throw new BuildFailedException("Missing bundled CPU inference file: "+name);Copy(Path.Combine(source,name),Path.Combine(destination,name));}
                if(!string.IsNullOrEmpty((string)descriptor["navigation_file"]))
                {
                    if((string)descriptor["navigation_file"]!="navigation.onnx")throw new BuildFailedException("Invalid exploration model filename");
                    Copy(Path.Combine(source,"navigation.onnx"),Path.Combine(destination,"navigation.onnx"));
                    using(var hash=System.Security.Cryptography.SHA256.Create())
                    {string actual=System.BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(Path.Combine(destination,"navigation.onnx")))).Replace("-","").ToLowerInvariant();if(actual!=(string)descriptor["navigation_sha256"])throw new BuildFailedException("Exploration model fingerprint mismatch");}
                }
                var export=Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(source,"export.json")));
                var crt=export["microsoft_crt"]?["files"] as Newtonsoft.Json.Linq.JObject;
                if(crt==null)throw new BuildFailedException("Export the app-local Microsoft CRT dependencies first.");
                foreach(var file in crt.Properties())
                {
                    if(Path.GetFileName(file.Name)!=file.Name||!file.Name.EndsWith(".dll",System.StringComparison.OrdinalIgnoreCase))throw new BuildFailedException("Invalid runtime filename");
                    string target=Path.Combine(destination,file.Name);Copy(Path.Combine(source,file.Name),target);
                    using(var hash=System.Security.Cryptography.SHA256.Create())
                    {string actual=System.BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(target))).Replace("-","").ToLowerInvariant();if(actual!=(string)file.Value)throw new BuildFailedException("Runtime fingerprint mismatch: "+file.Name);}
                }
                Copy(Path.Combine(source,"MICROSOFT-CRT-NOTICE.txt"),Path.Combine(destination,"MICROSOFT-CRT-NOTICE.txt"));
                using(var hash=System.Security.Cryptography.SHA256.Create())
                {string actual=System.BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(Path.Combine(destination,"volume.onnx")))).Replace("-","").ToLowerInvariant();if(actual!=(string)descriptor["volume_sha256"])throw new BuildFailedException("3D model fingerprint mismatch");}
            }
        }
    }
}
