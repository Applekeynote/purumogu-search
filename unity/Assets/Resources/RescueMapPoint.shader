Shader "Purumogu/MapPoint"
{
    Properties { _ViewportSize ("Viewport pixels", Vector) = (766,406,0,0) _PointScale ("Point scale",Float)=1 }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            float4 _ViewportSize;
            float _PointScale;
            struct Input { float4 vertex:POSITION;fixed4 color:COLOR;float3 uv:TEXCOORD0; };
            struct Output { float4 position:SV_POSITION;fixed4 color:COLOR;float2 corner:TEXCOORD0; };
            Output vert(Input v)
            {Output o;o.position=UnityObjectToClipPos(v.vertex);o.position.xy+=v.uv.xy*v.uv.z*_PointScale/_ViewportSize.xy*o.position.w;o.color=v.color;o.corner=v.uv.xy;return o;}
            fixed4 frag(Output i):SV_Target
            {float r=dot(i.corner,i.corner);clip(1-r);i.color.a*=1-smoothstep(.3,1,r);return i.color;}
            ENDCG
        }
    }
}
