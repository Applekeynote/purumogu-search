using UnityEngine;

namespace Purumogu
{
    static class DesktopWindow
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Initialize()
        {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
            if(Application.isBatchMode)return;
            // Override fullscreen preferences saved by earlier releases, once at startup.
            // Leave later user resizing and maximization to the normal Windows frame.
            var display=Screen.currentResolution;
            int width=Mathf.Min(1280,Mathf.FloorToInt(display.width*.85f));
            int height=Mathf.Min(720,Mathf.FloorToInt(display.height*.85f));
            Screen.SetResolution(Mathf.Max(320,width),Mathf.Max(180,height),FullScreenMode.Windowed);
#endif
        }
    }
}
