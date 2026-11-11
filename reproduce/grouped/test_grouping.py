import unittest
from pending_plan import make_plan


class GroupingPlanTest(unittest.TestCase):
    def test_only_missing_candidates_once_in_each_atom_bucket(self):
        manifest=[{'num_atoms':24},{'num_atoms':30},{'num_atoms':120},
                  {'num_atoms':400},{'num_atoms':900}]
        completed={(0,0),(0,3),(1,0),(1,1),(2,2),(3,0),(4,4)}
        caps={(0,96):8,(96,192):4,(384,512):4,(768,4096):1}
        plan=make_plan(range(5),manifest,completed,caps)
        planned=[x for group in plan for x in group]
        self.assertEqual(set(planned),{(i,c) for i in range(5) for c in range(5)}-completed)
        self.assertEqual(len(planned),len(set(planned)))
        for group in plan:
            sizes=[manifest[i]['num_atoms'] for i,_ in group]
            if max(sizes)<=96:self.assertLessEqual(len(group),8)
            elif max(sizes)<=192:self.assertLessEqual(len(group),4)
            elif max(sizes)<=512:self.assertLessEqual(len(group),4)
            else:self.assertEqual(len(group),1)
        self.assertTrue(any(len({i for i,_ in batch})>1 for batch in plan))
        self.assertNotIn((0,0),planned)

    def test_rejects_duplicate_assignments_and_unknown_completed_pairs(self):
        manifest=[{'num_atoms':24}]
        with self.assertRaises(ValueError):make_plan([0,0],manifest,set(),{})
        with self.assertRaises(ValueError):make_plan([0],manifest,{(1,0)},{})


if __name__=='__main__':unittest.main()
