template_program = '''
from typing import List

def select_next_job(
    partial_sequence: List[int],
    remaining_jobs: List[int],
    processing_times: List[List[int]],
) -> int:
    """Return one job id from remaining_jobs to append to the permutation."""
    return max(remaining_jobs, key=lambda job: sum(processing_times[job]))
'''

task_description = '''
Given a permutation flow-shop instance, construct a single job permutation used
on every machine so that the final makespan is minimized. Design a heuristic to
select the next job from the remaining jobs at each step.
'''
