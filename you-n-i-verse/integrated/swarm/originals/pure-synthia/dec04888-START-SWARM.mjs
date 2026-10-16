import { bootstrapCurrentSynthiaSwarm } from './src/synthia/swarm/bootstrap.mjs';
import { bindSelfhostedToSwarm } from './host/selfhostedSwarmHand.mjs';
const { swarm } = await bootstrapCurrentSynthiaSwarm();
bindSelfhostedToSwarm(swarm);
console.log(JSON.stringify(swarm.snapshot(), null, 2));
