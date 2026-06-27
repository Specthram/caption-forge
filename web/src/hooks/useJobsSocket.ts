/** Bridge the ``/ws/jobs`` WebSocket into the jobs store for the session. */

import { useEffect } from "react";
import { connectJobs } from "../api/ws";
import { useJobsStore } from "../store/jobsStore";

export function useJobsSocket() {
  const replaceAll = useJobsStore((state) => state.replaceAll);
  const upsert = useJobsStore((state) => state.upsert);

  useEffect(() => {
    return connectJobs((message) => {
      if (message.kind === "snapshot") replaceAll(message.jobs);
      else upsert(message.job);
    });
  }, [replaceAll, upsert]);
}
