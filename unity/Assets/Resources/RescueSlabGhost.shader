Shader "Purumogu/SlabGhost"
{
    Properties { _Color ("Slab tint", Color) = (.47,.63,.64,.22) }
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
            fixed4 _Color;
            struct VertexOutput { float4 position : SV_POSITION; };
            VertexOutput vert(float4 vertex : POSITION)
            {
                VertexOutput o; o.position=UnityObjectToClipPos(vertex); return o;
            }
            fixed4 frag(VertexOutput i) : SV_Target { return _Color; }
            ENDCG
        }
    }
}
