import unittest
from itertools import permutations
from cube_invariants import *

class CubeTests(unittest.TestCase):
    def test_isometry_exhaustive_small_cube(self):
        for perm in permutations(range(3)):
            for mask in range(8):
                op=CubeIsometry(perm,mask)
                for a in range(8):
                    self.assertEqual(op.inverse().apply(op.apply(a)),a)
                    for b in range(8):
                        self.assertEqual(hamming(a,b),hamming(op.apply(a),op.apply(b)))
    def test_composition(self):
        a=CubeIsometry((2,0,1),3); b=CubeIsometry((1,2,0),4)
        for s in range(8):self.assertEqual(a.compose(b).apply(s),a.apply(b.apply(s)))
    def test_minimum_five(self):
        result=verify_codebook((0,31),5,5)
        self.assertTrue(result['meets_constraint']);self.assertEqual(result['guaranteed_bit_error_correction'],2)
    def test_reject_close_states(self):
        self.assertFalse(verify_codebook((0,3),5,5)['meets_constraint'])
    def test_transform_preserves_distance(self):
        result=transform_codebook((0,31),CubeIsometry((4,3,2,1,0),7))
        self.assertTrue(result['identity_preserved']);self.assertEqual(result['after']['minimum_distance'],5)
    def test_invalid_transform(self):
        with self.assertRaises(ValueError):CubeIsometry((0,0,1))
    def test_invalid_codebook(self):
        with self.assertRaises(ValueError):verify_codebook((0,0),5)
    def test_five_axes_are_address_not_dof(self):
        self.assertEqual(five_axis_address(0b10101,(0,1,2,3,4)),dict(zip(('Movement','Evolution','Being','Design','Space'),(1,0,1,0,1))))
    def test_five_axes_reject_duplicates(self):
        with self.assertRaises(ValueError):five_axis_address(0,(0,1,2,3,3))
    def test_six_bit_state(self):
        op=CubeIsometry(tuple(range(6)),63)
        self.assertEqual(op.apply(0b101001),0b010110)

if __name__=='__main__':unittest.main()
