import torch
class PerStructureNoise:
 def __init__(self,sampler,entries):
  self.noise=sampler.diffusion_noise;self.original=self.noise.get_noise;self.entries=entries
  self.generators=[];self.states=[];self.calls=0
  outer=torch.cuda.get_rng_state()
  for e in entries:
   n=e['num_atoms']+8;torch.cuda.manual_seed(e['csp_seed'])
   # The native single-structure sampler consumes these two unused draws before get_sampling_start.
   torch.rand((1,n,1),device='cuda',dtype=torch.float32);torch.rand((1,n),device='cuda',dtype=torch.float32)
   self.states.append(torch.cuda.get_rng_state());self.generators.append(torch.Generator(device='cuda').manual_seed(e['csp_seed']))
  torch.cuda.set_rng_state(outer);self.noise.get_noise=self.get_noise
 def get_noise(self,positions,non_atom_mask,stable):
  result=torch.zeros_like(positions);outer=torch.cuda.get_rng_state()
  try:
   for j,e in enumerate(self.entries):
    end=e['num_atoms']+8;torch.cuda.set_rng_state(self.states[j]);self.noise.torch_generator=self.generators[j]
    result[j:j+1,:end]=self.original(positions[j:j+1,:end],non_atom_mask[j:j+1,:end],stable[j:j+1])
    self.states[j]=torch.cuda.get_rng_state()
   self.calls+=1
  finally:torch.cuda.set_rng_state(outer)
  return result
 def remove(self):self.noise.get_noise=self.original

