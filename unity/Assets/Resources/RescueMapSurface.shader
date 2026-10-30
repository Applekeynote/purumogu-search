Shader "Purumogu/ObservedSurface"
{
    Properties
    {
        _SurfaceGain ("Quiet depth backing",Range(0,1))=.12
        _WireColor ("Observed surface lattice",Color)=(.30,.86,.72,1)
        _GridWidth ("Grid line width in pixels",Range(.4,2))=.85
        _EchoEnabled ("Observation arrival highlight",Float)=1
        _MapNow ("Display clock",Float)=0
    }
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
            #pragma target 3.0
            #include "UnityCG.cginc"
            float _SurfaceGain,_GridWidth,_EchoEnabled,_MapNow;
            fixed4 _WireColor;
            struct Input {float4 vertex:POSITION;float3 normal:NORMAL;fixed4 color:COLOR;float3 grid:TEXCOORD0;};
            struct Output {float4 position:SV_POSITION;float3 normal:TEXCOORD0;fixed4 color:COLOR;float3 grid:TEXCOORD1;};
            Output vert(Input v){Output o;o.position=UnityObjectToClipPos(v.vertex);o.normal=UnityObjectToWorldNormal(v.normal);o.color=v.color;o.grid=v.grid;return o;}
            fixed4 frag(Output i):SV_Target
            {
                float light=.55+.45*saturate(dot(normalize(i.normal),normalize(float3(.35,.85,-.4))));
                float2 distanceToLine=abs(frac(i.grid.xy+.5)-.5)/max(fwidth(i.grid.xy),.0001);
                float gridLine=1-smoothstep(_GridWidth*.50,_GridWidth+ .45,min(distanceToLine.x,distanceToLine.y));
                float age=max(0,_MapNow-i.grid.z);
                float arrival=_EchoEnabled*saturate(1-age/1.15);
                float reveal=smoothstep(0,.3,age);
                fixed3 wire=lerp(_WireColor.rgb,fixed3(.88,1,.97),arrival*.75)*light;
                fixed3 backing=i.color.rgb*light*_SurfaceGain;
                return fixed4(lerp(backing,wire,gridLine*reveal),1);
            }
            ENDCG
        }
    }
}
