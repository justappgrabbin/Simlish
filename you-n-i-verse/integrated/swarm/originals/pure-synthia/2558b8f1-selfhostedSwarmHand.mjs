import { probeSelfhosted, selfhostedProcess, studioUrl } from './selfhostedBridge.js';

export function bindSelfhostedToSwarm(swarm) {
  const worker = swarm.registerExternal({
    id: 'hand:selfhosted-linux',
    group: 'linux-hands',
    location: '127.0.0.1:3000',
    maxConcurrency: 4,
    capabilities: ['selfhosted.probe','selfhosted.process','selfhosted.dashboard'],
    execute: async (task = {}) => {
      if (task.op === 'probe') return probeSelfhosted(task.timeoutMs);
      if (task.op === 'process') return selfhostedProcess(task.text || '', task.fallback);
      if (task.op === 'dashboard') return { url: studioUrl() };
      throw new Error(`Unknown selfhosted op: ${task.op}`);
    },
  });
  return Object.freeze({ registered: worker.id });
}

export default bindSelfhostedToSwarm;
