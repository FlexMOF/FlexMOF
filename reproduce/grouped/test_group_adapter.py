import unittest
import torch
from group_seq_adapter import GroupedSeedSequenceReuse


class FakeEncoder:
    training=False
    use_rope=False
    def __init__(self):self.seen=[]
    def forward(self,x,pair,batch,periodic,padding):
        self.seen.append(dict(packed=periodic['packed_edges'].tolist(),
                              distance=periodic['edge_distance'].tolist(),
                              pos=batch['pos'][:,0,0].tolist()))
        return x*2,pair.permute(2,0,1,3)*3


class GroupAdapterTest(unittest.TestCase):
    def test_only_within_structure_and_rebases_periodic_edges(self):
        pairs=[(101,1),(101,2),(202,0),(202,2),(202,4),(303,3)]
        module=FakeEncoder();f=GroupedSeedSequenceReuse(module,pairs)
        n=len(pairs);x=torch.arange(n,dtype=torch.float32)[None,:,None].expand(3,-1,2).clone()
        pair=torch.arange(n,dtype=torch.float32)[None,None,:,None].expand(3,3,-1,1).clone()
        batch={'pos':torch.arange(n,dtype=torch.float32)[:,None,None].expand(-1,3,3).clone()}
        periodic={'streaming_per_atom':True,'max_num_atoms':3,'candidate_count':4,
                  'edge_chunk_size':100,'cutoff':40.,
                  'packed_edges':torch.tensor([b*12+k for b in range(n) for k in (0,2)]),
                  'edge_distance':torch.tensor([10*b+k for b in range(n) for k in (0,2)],dtype=torch.float32)}
        with torch.inference_mode():a,b=f(x,pair,batch,periodic,torch.zeros(n,3,dtype=torch.bool))
        self.assertEqual(a[0,:,0].tolist(),[0,0,4,4,4,10])
        self.assertEqual(b[:,0,0,0].tolist(),[0,0,6,6,6,15])
        self.assertEqual(module.seen[0]['packed'],[0,2,12,14,24,26])
        self.assertEqual(module.seen[0]['distance'],[0,2,20,22,50,52])
        self.assertEqual(module.seen[0]['pos'],[0,2,5])
        self.assertEqual(f.stats()['reused_candidate_forwards'],3)

    def test_never_silently_accepts_duplicate_pair_or_training_mode(self):
        with self.assertRaises(ValueError):GroupedSeedSequenceReuse(FakeEncoder(),[(1,1),(1,1)])
        module=FakeEncoder();module.training=True;f=GroupedSeedSequenceReuse(module,[(1,1)])
        with self.assertRaises(RuntimeError):
            with torch.inference_mode():f(None,None,None,None,None)


if __name__=='__main__':unittest.main()
