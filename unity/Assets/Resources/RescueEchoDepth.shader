Shader "Purumogu/EchoDepthOnly"
{
    SubShader
    {
        Tags { "Queue"="Geometry" }
        Cull Off
        ColorMask 0
        ZWrite On
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            float4 vert(float4 vertex:POSITION):SV_POSITION{return UnityObjectToClipPos(vertex);}
            fixed4 frag():SV_Target{return 0;}
            ENDCG
        }
    }
}
