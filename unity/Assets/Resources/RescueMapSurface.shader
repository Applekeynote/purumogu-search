Shader "Purumogu/ObservedSurface"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Cull Off
        ZWrite On
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            struct Input {float4 vertex:POSITION;float3 normal:NORMAL;fixed4 color:COLOR;};
            struct Output {float4 position:SV_POSITION;float3 normal:TEXCOORD0;fixed4 color:COLOR;};
            Output vert(Input v){Output o;o.position=UnityObjectToClipPos(v.vertex);o.normal=UnityObjectToWorldNormal(v.normal);o.color=v.color;return o;}
            fixed4 frag(Output i):SV_Target
            {float light=.42+.58*saturate(dot(normalize(i.normal),normalize(float3(.35,.85,-.4))));return fixed4(i.color.rgb*light,1);}
            ENDCG
        }
    }
}
