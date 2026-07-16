/** Bridge the ``/ws/jobs`` WebSocket into the jobs store for the session. */

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { connectJobs } from "../api/ws";
import { useJobsStore } from "../store/jobsStore";

const TERMINAL = new Set(["done", "error", "stopped"]);

export function useJobsSocket() {
  const replaceAll = useJobsStore((state) => state.replaceAll);
  const upsert = useJobsStore((state) => state.upsert);
  const client = useQueryClient();

  useEffect(() => {
    return connectJobs((message) => {
      if (message.kind === "snapshot") {
        replaceAll(message.jobs);
        return;
      }
      upsert(message.job);

      // When a review job reaches a terminal state, refresh the review queue
      // centrally — the ReviewView tracker only covers runs launched from its
      // own button, so a review-after-generate run, or one the user has since
      // navigated away from, would otherwise leave the Review tab stale.
      const job = message.job;
      if (job.type === "review" && TERMINAL.has(job.state)) {
        client.invalidateQueries({ queryKey: ["review-findings"] });
        client.invalidateQueries({ queryKey: ["review-counts"] });
        client.invalidateQueries({ queryKey: ["caption-grid"] });
      }
    });
  }, [replaceAll, upsert, client]);
}
