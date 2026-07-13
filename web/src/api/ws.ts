/** Job-queue WebSocket client with auto-reconnect. */

import type { JobSnapshot } from "./types";

type JobsMessage =
  | { kind: "snapshot"; jobs: JobSnapshot[] }
  | { kind: "job"; job: JobSnapshot };

/**
 * Open the ``/ws/jobs`` socket and invoke ``onMessage`` for every event.
 * Returns a disposer that closes the socket and stops reconnecting.
 */
export function connectJobs(
  onMessage: (message: JobsMessage) => void,
): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws/jobs`);
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as JobsMessage);
      } catch {
        /* ignore malformed frames */
      }
    };
    socket.onclose = () => {
      if (!closed) retry = setTimeout(open, 1500);
    };
  };

  open();

  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    socket?.close();
  };
}
