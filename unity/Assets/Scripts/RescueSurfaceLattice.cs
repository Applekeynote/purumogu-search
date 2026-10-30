using System;
using System.Collections.Generic;
using UnityEngine;

namespace Purumogu.Rescue
{
    /// A display surface, built exclusively from measured occupied cells.
    /// Shared corners round by at most 22% of a cell, within observed space.
    /// The inference grid, free space, and evaluation geometry are never modified.
    public sealed class RescueSurfaceLattice
    {
        public sealed class Face
        {
            public Vector3Int Cell;
            public int Axis,Sign;
            public readonly Vector3Int[] Corners=new Vector3Int[4];
        }
        public readonly List<Face> Faces=new List<Face>();
        public readonly Dictionary<Vector3Int,Vector3> Positions=new Dictionary<Vector3Int,Vector3>();
        public readonly Dictionary<Vector3Int,Vector3> Normals=new Dictionary<Vector3Int,Vector3>();
        public float MaximumShift{get;private set;}
        static readonly Vector2Int[] Corners={new Vector2Int(-1,-1),new Vector2Int(-1,1),new Vector2Int(1,1),new Vector2Int(1,-1)};
        public RescueSurfaceLattice(HashSet<Vector3Int> occupied,float cell,Func<Vector3,bool> observed)
        {
            var neighbors=new Dictionary<Vector3Int,HashSet<Vector3Int>>();
            foreach(var k in occupied)for(int axis=0;axis<3;axis++)for(int sign=-1;sign<=1;sign+=2)
            {
                var outside=k;outside[axis]+=sign;if(occupied.Contains(outside))continue;
                var face=new Face{Cell=k,Axis=axis,Sign=sign};var normal=Vector3.zero;normal[axis]=sign;
                int a=(axis+1)%3,b=(axis+2)%3;
                for(int i=0;i<4;i++)
                {
                    var corner=k;corner[axis]+=sign>0?1:0;corner[a]+=Corners[i].x>0?1:0;corner[b]+=Corners[i].y>0?1:0;
                    face.Corners[i]=corner;
                    if(!neighbors.ContainsKey(corner)){neighbors[corner]=new HashSet<Vector3Int>();Normals[corner]=Vector3.zero;}
                    Normals[corner]+=normal;
                }
                for(int i=0;i<4;i++)
                {var from=face.Corners[i];var to=face.Corners[(i+1)%4];neighbors[from].Add(to);neighbors[to].Add(from);}
                Faces.Add(face);
            }
            foreach(var pair in neighbors)
            {
                var original=(Vector3)pair.Key*cell;var average=Vector3.zero;
                foreach(var other in pair.Value)average+=(Vector3)other*cell;
                average/=pair.Value.Count;
                var shift=Vector3.ClampMagnitude((average-original)*.65f,cell*.22f);
                var candidate=original+shift;
                if(observed!=null&&!observed(candidate))candidate=original;
                MaximumShift=Mathf.Max(MaximumShift,Vector3.Distance(original,candidate));
                Positions[pair.Key]=candidate;Normals[pair.Key]=Normals[pair.Key].normalized;
            }
        }
    }
}
