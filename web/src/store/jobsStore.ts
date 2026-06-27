/** Live job-queue state, fed by the ``/ws/jobs`` WebSocket. */

import { useMemo } from "react";
import { create } from "zustand";
import type { JobSnapshot } from "../api/types";

interface JobsState {
  jobs: Record<string, JobSnapshot>;
  upsert: (job: JobSnapshot) => void;
  replaceAll: (jobs: JobSnapshot[]) => void;
  clearFinished: () => void;
}

export const useJobsStore = create<JobsState>((set) => ({
  jobs: {},
  upsert: (job) =>
    set((state) => ({ jobs: { ...state.jobs, [job.id]: job } })),
  replaceAll: (jobs) =>
    set(() => ({
      jobs: Object.fromEntries(jobs.map((job) => [job.id, job])),
    })),
  clearFinished: () =>
    set((state) => ({
      jobs: Object.fromEntries(
        Object.entries(state.jobs).filter(
          ([, job]) => job.state === "queued" || job.state === "running",
        ),
      ),
    })),
}));

/**
 * Jobs sorted newest-first. Memoised over the (stable) jobs map so the
 * selector never returns a fresh array on every render — a new reference
 * each call would make Zustand's ``useSyncExternalStore`` loop.
 */
export function useJobList(): JobSnapshot[] {
  const jobs = useJobsStore((state) => state.jobs);
  return useMemo(
    () =>
      Object.values(jobs).sort((a, b) => b.created_at - a.created_at),
    [jobs],
  );
}

/** How many jobs are queued or running (a primitive, safe to select). */
export function useRunningCount(): number {
  return useJobsStore(
    (state) =>
      Object.values(state.jobs).filter(
        (job) => job.state === "queued" || job.state === "running",
      ).length,
  );
}
