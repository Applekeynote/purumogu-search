using System.IO;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Purumogu.Editor
{
    public sealed class BundleInferenceBuild : IPreprocessBuildWithReport
    {
        public int callbackOrder=>0;
        public void OnPreprocessBuild(BuildReport report)
        {
            string root=Path.GetFullPath(Path.Combine(Application.dataPath,"../.."));
            string server=Path.Combine(root,"rust/target/release/rescue_server.exe");
            string model=Path.Combine(root,"models/swarm-v34-local/model.json");
            if(!File.Exists(server)||!File.Exists(model))throw new BuildFailedException("Build rescue_server --release first; bundled model must exist.");
            string destination=Path.Combine(Application.streamingAssetsPath,"inference");Directory.CreateDirectory(destination);
            File.Copy(server,Path.Combine(destination,"rescue_server.exe"),true);
            File.Copy(model,Path.Combine(destination,"model.json"),true);
        }
    }
}
