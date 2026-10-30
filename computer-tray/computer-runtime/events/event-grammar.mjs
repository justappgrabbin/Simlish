/**
 * Computer's existing grammar-v1 event contract, shared by file-backed and
 * IndexedDB-backed event emitters. Unknown fields remain null, never invented.
 */
export const EVENT_TYPES = Object.freeze([
  'user_action', 'agent_action', 'message', 'project_action', 'experiment_action',
  'tool_execution', 'file_ingest', 'build', 'network_connection', 'state_transition',
  'graph_activation', 'application_launch', 'movement', 'observation', 'verification',
  'failure', 'recovery',
]);
export const ACTOR_TYPES = Object.freeze([
  'user', 'mirrored_agent', 'application', 'process', 'automaton', 'device', 'system',
]);
export function buildComputerEvent(partial = {}, { uuid = () => crypto.randomUUID() } = {}) {
  const event = {
    event_id: partial.event_id ?? 'evt-' + uuid(),
    timestamp: partial.timestamp ?? new Date().toISOString(),
    actor_id: partial.actor_id ?? 'system',
    actor_type: ACTOR_TYPES.includes(partial.actor_type) ? partial.actor_type : 'system',
    target_id: partial.target_id ?? null,
    event_type: EVENT_TYPES.includes(partial.event_type) ? partial.event_type : 'observation',
    input: partial.input ?? null,
    actor_address: partial.actor_address ?? null,
    target_address: partial.target_address ?? null,
    event_address: partial.event_address ?? null,
    pre_state: partial.pre_state ?? null,
    context: partial.context ?? null,
    process_id: partial.process_id ?? null,
    provider: partial.provider ?? null,
    address_provider: partial.address_provider ?? null,
    state_provider: partial.state_provider ?? null,
    result: partial.result ?? null,
    result_status: ['success', 'failure', 'partial', 'unknown'].includes(partial.result_status)
      ? partial.result_status : 'unknown',
    post_state: partial.post_state ?? null,
    observable_effect: partial.observable_effect ?? null,
    related_events: [...(partial.related_events ?? [])],
    parents: [...(partial.parents ?? [])],
    trajectory_id: partial.trajectory_id ?? 'traj-' + uuid(),
    evidence: partial.evidence ?? null,
  };
  return Object.freeze(event);
}
