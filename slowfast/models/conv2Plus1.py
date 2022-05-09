import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class Conv2Plus1(nn.Module):
  '''
  R(2+1) D convolution as decribed in 
  https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf
  Summary: Implement 3D (THW) convolution as 2d (HW) + 1D (T)
  
  Args:
  in_dim - Number of input channles
  out_dim - Number of output channel
  THW - triple giving the dimensions for T - time, H - Height - W - width
  kernel - Kernel size - either an integer or 3 dimension list (T, W, H)
  stride - Stride - either an integer or 3 dimension list
  padding - either an integer or 3 dimension list
  '''
  def __init__(self, in_dim, out_dim, THW, kernel_size, stride=1, padding=None, bias=True):
    super().__init__()
    (self.T, self.H, self.W) = THW
    self.in_dim = in_dim
    self.out_dim = out_dim

    # Expand out integer where necessary
    kernel_size = _triple(kernel_size)
    assert len(kernel_size) == 3, "Expected kernel to be 3 dimensional got {}".format(kernel_size)
    stride = _triple(stride)
    assert len(stride) == 3, "Expected stride to be 3 dimensional got {}".format(stride)
#     assert kernel[0] <= self.T , "Expect Timekernel size {} <= timeDim {}".format(kernel[0], self.T)

    # Intermediate channels = out_dim * time_dim
    intm_chans = self.out_dim * self.T
    s_kernel = kernel_size[1:]
    s_pad = [x // 2 for x in s_kernel]
    self.spatial = nn.Conv2d(self.in_dim*self.T, intm_chans, kernel_size=s_kernel, stride=stride[1:], padding=s_pad, bias=bias, groups=self.T)

    if self.T > 1:
      # Temporal is a 1 D but use 2D for performance. 
      # Use  the last dimension for the time Convolution
      t_kernel = [1, kernel_size[0]]
      t_pad = [x // 2 for x in t_kernel]
      t_stride = [1, stride[0]]
      wOut = self.W // stride[-1]
      t_chans = self.out_dim * wOut
      self.temporal = nn.Conv2d(t_chans, t_chans, kernel_size=t_kernel, stride=t_stride, padding=t_pad, bias=bias, groups=wOut)
    else:
      self.temporal = None
    
    # self.init_kernel(self.spatial)
    # self.init_kernel(self.temporal)
    

  def init_kernel(self, cc):
    for k, v in cc.named_parameters():
      print(k, v.shape)
      nn.init.constant_(v, 1)    
  
    
  def forward(self, x):
    # tensor is of shape 
    # B x C x T x H x W
    # B - batch, C - Input Chans
    
    # Step 1; Do the 2D conv
    # Change input to B x (TxC) x H x W
    # Note important to permute the T anc C channels so grouped conv works as expected
    dims = x.shape
    # print("Spatial {} X shape {}".format(self.spatial, x.shape))
    if self.T  == 1:
      # Single time dimension can collapse to just a conv2D
      x = torch.squeeze(x, dim=2)
      out = self.spatial(x)
      out = out.unsqueeze(2)
    else: 
      x = x.permute((0, 2, 1, 3, 4))
      x = x.reshape((dims[0], dims[1]*dims[2], dims[3], dims[4]))
      intm = self.spatial(x)
      self.intt = intm
      
      # Step 2 - Do the 1D conv. using a 2D conv
      # Reshape the intermediate tensor so Time is on lowest dimension 
      intm = intm.reshape((dims[0], dims[2], -1, intm.shape[2], intm.shape[3]))
      intm = intm.permute(0, 4, 2, 3, 1)
      dimsIntm = intm.shape
      intm = intm.reshape(-1, intm.shape[1]*intm.shape[2], intm.shape[3], intm.shape[4])
      # print("Temporal {} inpt {} ".format(self.temporal, intm.shape))
      out = self.temporal(intm)
      
      # print("Atfer Temp {}".format(out.shape))
      #  Finally reshape the output back to B x C x T x H x W
      out = out.reshape(-1, dimsIntm[1], self.out_dim, out.shape[2], out.shape[3])
      out = out.permute((0, 2, 4, 3, 1))

    # print("OUT dims {}".format(out.shape))
    return out
