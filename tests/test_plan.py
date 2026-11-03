import random
import unittest

from sojupscale import plan


class SplitShot(unittest.TestCase):
    def test_short_shot_is_one_piece(self):
        self.assertEqual(plan.split_shot(10, 50, 100, 8), [(10, 50)])

    def test_long_shot_overlaps_exactly(self):
        for length in range(101, 2000, 37):
            parts = plan.split_shot(5, 5 + length, 100, 12)
            self.assertEqual(parts[0][0], 5)
            self.assertEqual(parts[-1][1], 5 + length)
            for (s0, e0), (s1, e1) in zip(parts, parts[1:]):
                self.assertEqual(e0 - s1, 12)
            self.assertTrue(all(e - s <= 100 for s, e in parts), parts)
            sizes = [e - s for s, e in parts]
            self.assertLessEqual(max(sizes) - min(sizes), 2, sizes)

    def test_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            plan.split_shot(0, 500, 10, 10)


class BalancedBatch(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(plan.balanced_batch(30, 45, 3), 45)    # one batch anyway
        self.assertEqual(plan.balanced_batch(100, 45, 3), 37)   # 3 batches 37/37/32, not 45/45/16
        self.assertEqual(plan.batch_starts(100, 37, 3), [0, 34, 68])
        self.assertEqual(plan.batch_starts(100, 45, 3), [0, 42, 84])

    def test_never_more_batches_and_no_runt(self):
        for overlap in (0, 3, 8):
            for max_batch in (21, 45, 81):
                for n in range(1, 700):
                    b = plan.balanced_batch(n, max_batch, overlap)
                    self.assertLessEqual(b, max_batch)
                    plain = plan.batch_starts(n, max_batch, overlap)
                    bal = plan.batch_starts(n, b, overlap)
                    self.assertEqual(len(bal), len(plain), (n, max_batch, overlap))
                    # the last batch is never shorter than without balancing
                    self.assertGreaterEqual(n - bal[-1], n - plain[-1], (n, b, overlap))

    def test_typical_lengths_lose_the_runt(self):
        # 45-frame batches, overlap 3: the shortest batch of a typical shot grows a lot
        worst_plain = worst_bal = 45
        for n in range(46, 1000):
            b = plan.balanced_batch(n, 45, 3)
            worst_plain = min(worst_plain, n - plan.batch_starts(n, 45, 3)[-1])
            worst_bal = min(worst_bal, n - plan.batch_starts(n, b, 3)[-1])
        self.assertLessEqual(worst_plain, 5)
        self.assertGreaterEqual(worst_bal, 20)


class PlanJobs(unittest.TestCase):
    def assertValid(self, jobs, first, last, cuts):
        plan.check(jobs, first, last, cuts)

    def test_cuts_become_job_or_piece_boundaries(self):
        cuts = [100, 250, 260, 900]
        jobs = plan.plan_jobs(0, 1000, cuts, job_frames=600, max_piece=300, overlap=12)
        self.assertValid(jobs, 0, 1000, cuts)
        starts = {p["start"] for j in jobs for p in j}
        self.assertTrue({100, 250, 260, 900} <= starts)

    def test_long_take_blends_inside_jobs_and_cuts_between(self):
        jobs = plan.plan_jobs(0, 5000, [], job_frames=1200, max_piece=600, overlap=12)
        self.assertValid(jobs, 0, 5000, [])
        self.assertGreater(len(jobs), 3)
        for job in jobs:
            for prev, nxt in zip(job, job[1:]):
                self.assertEqual(prev["blend_out"], 12)
            self.assertLessEqual(sum(len(plan.written_frames(p)) for p in job) + 12, 1200 + 12)
        # across jobs: hard cut in the middle of the overlap, context on both sides
        for a, b in zip(jobs, jobs[1:]):
            prev, nxt = a[-1], b[0]
            self.assertEqual(prev["keep_end"], nxt["keep_start"])
            self.assertGreater(prev["end"], prev["keep_end"])
            self.assertLess(nxt["start"], nxt["keep_start"])

    def test_random_plans_are_valid(self):
        rng = random.Random(1)
        for _ in range(300):
            first = rng.randrange(0, 500)
            last = first + rng.randrange(1, 8000)
            cuts = sorted(rng.sample(range(first, last + 1), min(rng.randrange(0, 40), last - first)))
            max_piece = rng.randrange(20, 700)
            overlap = rng.randrange(0, min(24, max_piece // 4) + 1)
            job_frames = max_piece + rng.randrange(0, 2000)
            jobs = plan.plan_jobs(first, last, cuts, job_frames, max_piece, overlap)
            self.assertValid(jobs, first, last, cuts)
            for job in jobs:
                self.assertTrue(all(p["end"] - p["start"] <= max_piece for p in job))
                lo, hi = plan.job_range(job)
                self.assertLessEqual(hi - lo, job_frames + overlap)

    def test_rejects_overlap_too_large_for_crossfades(self):
        with self.assertRaises(ValueError):
            plan.plan_jobs(0, 1000, [], 600, 100, 26)

    def test_check_catches_gaps(self):
        jobs = plan.plan_jobs(0, 300, [], 300, 300, 0)
        jobs[0][0]["keep_end"] = 299
        with self.assertRaises(ValueError):
            plan.check(jobs, 0, 300, [])


if __name__ == "__main__":
    unittest.main()
