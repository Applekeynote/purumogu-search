using UnityEngine;

namespace Purumogu.Rescue
{
    public sealed partial class RescueController
    {
        // The upper-right pane is only a green wire map, with no HUD or inset.
        static readonly Rect MapRenderArea=new Rect(810,126,766,446);
        void ApplyMapCamera()
        {inferenceView.Apply(mapCamera,new Vector3(0,.24f,1.4f),new Vector3(2.35f,2.9f,-3.2f),1.20f);}
    }
}
