Shader "Purumogu/QuietGreenEchoWire"
{
    Properties
    {
        _ViewportSize ("Viewport pixels",Vector)=(766,446,0,0)
        _WireColor ("Green wire",Color)=(1,1,1,1)
        _HalfWidth ("Half line width in pixels",Float)=.70
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off
        ZWrite Off
        Offset -1,-1
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            float4 _ViewportSize;
            fixed4 _WireColor;
            float _HalfWidth;
            struct Input {float4 vertex:POSITION;float3 tangent:NORMAL;float2 side:TEXCOORD0;float2 estimate:TEXCOORD1;fixed4 color:COLOR;};
            struct Output {float4 position:SV_POSITION;float side:TEXCOORD0;float depth:TEXCOORD1;float along:TEXCOORD2;float estimate:TEXCOORD3;fixed4 color:COLOR;};
            Output vert(Input v)
            {
                Output o;float4 start=UnityObjectToClipPos(v.vertex),end=UnityObjectToClipPos(float4(v.vertex.xyz+v.tangent,1));
                float2 delta=(end.xy/end.w-start.xy/start.w)*_ViewportSize.xy;
                float2 perpendicular=float2(-delta.y,delta.x)/max(length(delta),.00001);
                o.position=start;o.position.xy+=perpendicular*v.side.x*_HalfWidth*2/_ViewportSize.xy*start.w;
                o.side=v.side.x;o.along=v.side.y;o.estimate=v.estimate.x;o.color=v.color;o.depth=-UnityObjectToViewPos(v.vertex).z;return o;
            }
            fixed4 frag(Output i):SV_Target
            {
                if(i.estimate>.5)clip(frac(i.along/.018)-.28);
                float opacity=1-smoothstep(.32,1,abs(i.side));
                float depthShade=lerp(1,.50,saturate((i.depth-3)/5));
                return fixed4(_WireColor.rgb*i.color.rgb*depthShade,_WireColor.a*i.color.a*opacity);
            }
            ENDCG
        }
    }
}
