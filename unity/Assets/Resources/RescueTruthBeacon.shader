Shader "Purumogu/TruthBeacon"
{
    SubShader
    {
        Tags { "Queue"="Overlay" }
        ZTest Always
        ZWrite Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            struct V{float4 vertex:POSITION;float3 normal:NORMAL;};
            struct O{float4 vertex:SV_POSITION;float3 normal:TEXCOORD0;};
            O vert(V v){O o;o.vertex=UnityObjectToClipPos(v.vertex);o.normal=UnityObjectToWorldNormal(v.normal);return o;}
            fixed4 frag(O i):SV_Target {float light=.5+.5*saturate(dot(normalize(i.normal),normalize(float3(-.3,.8,-.6))));return fixed4(light,.035,.055,1);}
            ENDCG
        }
    }
}
