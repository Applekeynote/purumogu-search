"""3D geometry networks from relative observed evidence, without scene or label inputs."""
import torch
from torch import nn
from torch.nn import functional as F


def rotate_evidence(x,turns):
    """Rotate spatial axes AND relative endpoint/direction components around gravity."""
    for _ in range(turns%4):
        x=torch.rot90(x,1,(2,4))
        if x.shape[1] in (9,18,74,90):
            values=x[:,[3,5,6,8]].clone()
            x[:,3]=-values[:,1];x[:,5]=values[:,0]
            x[:,6]=-values[:,3];x[:,8]=values[:,2]
        if x.shape[1]==18:
            covariance=x[:,[9,11,12,13,14,15,17]].clone()
            x[:,9]=covariance[:,1];x[:,11]=covariance[:,0]
            x[:,12]=-covariance[:,4];x[:,13]=-covariance[:,3];x[:,14]=covariance[:,2]
            x[:,15]=covariance[:,6];x[:,17]=covariance[:,5]
        if x.shape[1] in (74,90):
            fine=x[:,10:74].clone()
            for octant in range(8):
                bx,by,bz=(octant>>2)&1,(octant>>1)&1,octant&1
                destination=(1-bz)*4+by*2+bx
                block=fine[:,octant*8:(octant+1)*8].clone()
                block[:,1]=-fine[:,octant*8+3];block[:,3]=fine[:,octant*8+1]
                block[:,4]=-fine[:,octant*8+6];block[:,6]=fine[:,octant*8+4]
                x[:,10+destination*8:18+destination*8]=block
            if x.shape[1]==90:
                free=x[:,74:90].clone()
                for octant in range(8):
                    bx,by,bz=(octant>>2)&1,(octant>>1)&1,octant&1
                    destination=(1-bz)*4+by*2+bx
                    x[:,74+destination*2:76+destination*2]=free[:,octant*2:octant*2+2]
    return x


def reflect_evidence(x,axis):
    x=x.flip(axis)
    if x.shape[1] in (9,18,74,90):
        for feature in ([3,6] if axis==2 else [5,8]):x[:,feature]=-x[:,feature]
    if x.shape[1]==18:
        for feature in ([12,13] if axis==2 else [13,14]):x[:,feature]=-x[:,feature]
    if x.shape[1] in (74,90):
        fine=x[:,10:74].clone()
        for octant in range(8):
            destination=octant^(4 if axis==2 else 1)
            block=fine[:,octant*8:(octant+1)*8].clone()
            for feature in ([1,4] if axis==2 else [3,6]):block[:,feature]=-block[:,feature]
            x[:,10+destination*8:18+destination*8]=block
        if x.shape[1]==90:
            free=x[:,74:90].clone()
            for octant in range(8):
                destination=octant^(4 if axis==2 else 1)
                x[:,74+destination*2:76+destination*2]=free[:,octant*2:octant*2+2]
    return x


def tiled_logits(model,x):
    result=torch.zeros((x.shape[0],1,32,24,72),device=x.device,dtype=x.dtype)
    counts=torch.zeros_like(result)
    # Fixed overlapping local windows, identical at validation and standalone inference.
    for y in [0,8]:
        for z in [0,16,32,40]:
            value=model(x[:,:,:,y:y+16,z:z+32])
            result[:,:,:,y:y+16,z:z+32]+=value
            counts[:,:,:,y:y+16,z:z+32]+=1
    return result/counts


class ChannelNorm(nn.LayerNorm):
    def forward(self,x):
        return super().forward(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)


class Block(nn.Sequential):
    def __init__(self, a, b,normalization='group'):
        def norm():return nn.BatchNorm3d(b,momentum=.05) if normalization=='batch' else ChannelNorm(b) if normalization=='channel' else nn.GroupNorm(4,b)
        super().__init__(nn.Conv3d(a,b,3,padding=1,bias=False),norm(),nn.ReLU(),
                         nn.Conv3d(b,b,3,padding=1,bias=False),norm(),nn.ReLU())


class PlaneProfile(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv1d(8,16,3,padding=1),nn.ReLU(),nn.Conv1d(16,16,3,padding=2,dilation=2),
                         nn.ReLU(),nn.Conv1d(16,8,3,padding=4,dilation=4),nn.ReLU())


def plane_statistics(x,axis):
    """Global measured-plane consensus, indexed by relative grid layers; no plane heights supplied."""
    reduced=tuple(d for d in (2,3,4) if d!=axis)
    hit=x[:,2:3].clamp(min=0,max=1);den=hit.sum(reduced).clamp(min=1e-6)
    offset=x[:,axis+1:axis+2];direction=x[:,axis+4:axis+5]
    observed=(x[:,1:2]>0).to(x.dtype)
    return torch.cat((x[:,0:1].mean(reduced),x[:,1:2].mean(reduced),observed.mean(reduced),hit.mean(reduced),
                      (hit*offset).sum(reduced)/den,(hit*direction).sum(reduced)/den,
                      (hit*direction.abs()).sum(reduced)/den,(observed*(x[:,0:1]>0)).mean(reduced)),1)


class VolumeNet(nn.Module):
    def __init__(self,width=16,residual=False,channels=2,profiles=False,normalization='group'):
        super().__init__()
        self.profiles=profiles
        if profiles:
            if channels<9:raise ValueError('Measured range moments are required for plane consensus')
            self.horizontal=PlaneProfile();self.vertical=PlaneProfile()
        self.e0=Block(channels+(24 if profiles else 0),width,normalization);self.e1=Block(width,width*2,normalization);self.e2=Block(width*2,width*4,normalization);self.mid=Block(width*4,width*6,normalization)
        self.d2=Block(width*10,width*4,normalization);self.d1=Block(width*6,width*2,normalization);self.d0=Block(width*3,width,normalization)
        self.out=nn.Conv3d(width,1,1);self.residual=residual
        if residual:self.sensor_gain=nn.Parameter(torch.tensor(.5))

    def features(self,x):
        if self.profiles:
            px=self.horizontal(plane_statistics(x,2))[:,:,:,None,None].expand(-1,-1,-1,x.shape[3],x.shape[4])
            py=self.vertical(plane_statistics(x,3))[:,:,None,:,None].expand(-1,-1,x.shape[2],-1,x.shape[4])
            pz=self.horizontal(plane_statistics(x,4))[:,:,None,None,:].expand(-1,-1,x.shape[2],x.shape[3],-1)
            x=torch.cat((x,px,py,pz),1)
        a=self.e0(x);b=self.e1(F.avg_pool3d(a,2));c=self.e2(F.avg_pool3d(b,2));d=self.mid(F.avg_pool3d(c,2))
        d=self.d2(torch.cat((F.interpolate(d,scale_factor=2,mode='nearest'),c),1))
        d=self.d1(torch.cat((F.interpolate(d,scale_factor=2,mode='nearest'),b),1))
        d=self.d0(torch.cat((F.interpolate(d,scale_factor=2,mode='nearest'),a),1))
        return d

    def forward(self,x):
        return self.out(self.features(x))+(self.sensor_gain*x[:,0:1]*4 if self.residual else 0)


class LocalSurfaceBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.spatial=nn.Conv3d(width,width,3,padding=dilation,dilation=dilation,groups=width)
        # Normalize channels at each location. Spatial GroupNorm would make a
        # nominally local model depend on the entire building's occupancy.
        self.norm=ChannelNorm(width)
        self.expand=nn.Conv3d(width,width*3,1);self.project=nn.Conv3d(width*3,width,1)
        self.gain=nn.Parameter(torch.full((1,width,1,1,1),.1))

    def forward(self,x):
        return x+self.gain*self.project(F.gelu(self.expand(self.norm(self.spatial(x)))))


class LocalSurfaceNet(nn.Module):
    """15-cell receptive field; no global shape pooling or absolute coordinates."""
    def __init__(self,width=32,residual=True,channels=74):
        super().__init__();self.residual=residual
        self.input=nn.Conv3d(channels,width,1)
        self.blocks=nn.Sequential(*(LocalSurfaceBlock(width,d) for d in [1,1,2,2,1]))
        self.out=nn.Conv3d(width,1,1)
        if residual:self.sensor_gain=nn.Parameter(torch.tensor(.5))

    def forward(self,x):
        return self.out(self.blocks(self.input(x)))+(self.sensor_gain*x[:,0:1]*4 if self.residual else 0)


def lift_octants(x,features):
    """Interleave bx/by/bz octants into a real 2.5cm spatial grid."""
    b,_,sx,sy,sz=x.shape
    return x.reshape(b,2,2,2,features,sx,sy,sz).permute(0,4,5,1,6,2,7,3).reshape(b,features,sx*2,sy*2,sz*2)


def fold_octants(x):
    """Keep all eight fine spatial phases as channels, without average pooling."""
    b,c,sx,sy,sz=x.shape
    if sx%2 or sy%2 or sz%2:raise ValueError('Even fine-grid dimensions required')
    return x.reshape(b,c,sx//2,2,sy//2,2,sz//2,2).permute(0,3,5,7,1,2,4,6).reshape(b,8*c,sx//2,sy//2,sz//2)


def extend_correction_inputs(previous,model,additional_channels):
    """Function-preserving zero initialization for a strictly appended feature block."""
    previous=dict(previous);old=previous['correction.0.weight']
    new=torch.zeros_like(model.state_dict()['correction.0.weight'])
    if new.shape[1]!=old.shape[1]+additional_channels or new.shape[0]!=old.shape[0] or new.shape[2:]!=old.shape[2:]:raise ValueError('Correction extension contract differs')
    new[:,:old.shape[1]]=old;previous['correction.0.weight']=new
    return previous


class SubvoxelRefinementNet(nn.Module):
    """Retain global completion and learn local fine-surface corrections."""
    def __init__(self,width=24,residual=True,channels=90,profiles=True,normalization='group',backbone_channels=74,fine_width=12,freeze_backbone=True,fine_plane_profiles=False,fine_supervision=False,fine_unet=False,fine_occupancy_phases=False):
        super().__init__()
        if channels not in (74,90) or backbone_channels not in (74,90) or backbone_channels>channels:raise ValueError('Fine-octant inputs required')
        self.backbone=VolumeNet(width,residual,backbone_channels,profiles,normalization)
        self.backbone_channels=backbone_channels;self.freeze_backbone=freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():parameter.requires_grad_(False)
        self.fine_unet=fine_unet
        if fine_occupancy_phases and not (fine_unet and fine_supervision):raise ValueError('Fine occupancy phases require the supervised fine U-Net')
        self.fine_occupancy_phases=fine_occupancy_phases
        if fine_unet:
            self.fine_backbone=VolumeNet(fine_width,False,12 if channels==90 else 10,False,normalization)
            del self.fine_backbone.out
        else:
            self.fine_input=nn.Conv3d(12 if channels==90 else 10,fine_width,3,padding=1)
            self.fine_blocks=nn.Sequential(LocalSurfaceBlock(fine_width,1),LocalSurfaceBlock(fine_width,2))
        if fine_supervision:self.fine_out=nn.Conv3d(fine_width,1,1)
        self.fine_logits=None
        self.fine_plane_profiles=fine_plane_profiles
        if fine_plane_profiles:self.fine_plane_profile=PlaneProfile()
        self.correction=nn.Sequential(nn.Conv3d(fine_width*2+3+(16 if fine_plane_profiles else 0)+(8 if fine_occupancy_phases else 0),16,3,padding=1),nn.GELU(),nn.Conv3d(16,1,1))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_backbone:self.backbone.eval()
        return self

    def forward(self,x):
        if self.freeze_backbone:
            with torch.no_grad():baseline=self.backbone(x[:,:self.backbone_channels])
        else:baseline=self.backbone(x[:,:self.backbone_channels])
        parts=[lift_octants(x[:,10:74],8),F.interpolate(x[:,:2],scale_factor=2,mode='nearest')]
        if x.shape[1]==90:parts.append(lift_octants(x[:,74:90],2))
        measured=torch.cat(parts,1)
        fine=self.fine_backbone.features(measured) if self.fine_unet else self.fine_blocks(F.gelu(self.fine_input(measured)))
        fine_logits=self.fine_out(fine) if hasattr(self,'fine_out') and (self.training or self.fine_occupancy_phases) else None
        self.fine_logits=fine_logits if self.training else None
        pooled=[F.avg_pool3d(fine,2),F.max_pool3d(fine,2),x[:,:2],baseline]
        if self.fine_plane_profiles:
            weight=measured[:,0:1].clamp(min=0,max=1);den=weight.sum((2,4)).clamp(min=1e-6)
            sdf=measured[:,10:11] if x.shape[1]==90 else torch.zeros_like(weight)
            support=measured[:,11:12] if x.shape[1]==90 else torch.zeros_like(weight)
            stats=torch.cat((weight.mean((2,4)),measured[:,7:8].mean((2,4)),(weight>0).to(weight.dtype).mean((2,4)),
                (weight*measured[:,2:3]).sum((2,4))/den,(weight*measured[:,5:6]).sum((2,4))/den,
                (weight*measured[:,5:6].abs()).sum((2,4))/den,support.mean((2,4)),
                (sdf*support).sum((2,4))/support.sum((2,4)).clamp(min=1e-6)),1)
            profile=self.fine_plane_profile(stats)+stats
            # Preserve both 2.5cm height phases inside each 5cm output cell.
            profile=profile.reshape(x.shape[0],8,x.shape[3],2).permute(0,1,3,2).reshape(x.shape[0],16,x.shape[3])
            pooled.append(profile[:,:,None,:,None].expand(-1,-1,x.shape[2],-1,x.shape[4]))
        if self.fine_occupancy_phases:pooled.append(fold_octants(fine_logits.clamp(-12,12)))
        pooled=torch.cat(pooled,1)
        return baseline+self.correction(pooled)


def build_network(state):
    if state.get('architecture') in ('subvoxel-refinement-v1','subvoxel-plane-refinement-v2','subvoxel-unet-refinement-v3','subvoxel-phase-refinement-v4'):
        return SubvoxelRefinementNet(state.get('width',24),state.get('residual',True),state.get('channels',90),state.get('profiles',True),state.get('normalization','group'),state.get('backbone_channels',74),state.get('fine_width',12),state.get('freeze_backbone',True),state.get('architecture') in ('subvoxel-plane-refinement-v2','subvoxel-unet-refinement-v3','subvoxel-phase-refinement-v4'),state.get('fine_supervision',False),state.get('architecture') in ('subvoxel-unet-refinement-v3','subvoxel-phase-refinement-v4'),state.get('architecture')=='subvoxel-phase-refinement-v4')
    if state.get('architecture','volume-unet-v1')=='local-surface-v1':
        return LocalSurfaceNet(state.get('width',32),state.get('residual',True),state.get('channels',74))
    if state.get('architecture','volume-unet-v1')!='volume-unet-v1':raise ValueError('Unknown 3D architecture')
    return VolumeNet(state.get('width',16),state.get('residual',False),state.get('channels',2),state.get('profiles',False),state.get('normalization','group'))
