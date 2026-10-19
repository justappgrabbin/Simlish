// Gaming area host. WRAPS her youniverse game app (vendor/youniverse-games, byte-identical, see PROVENANCE.txt) — her
// WorldAdapter + PlayerSeedStore + GameStateStore + MorphEngine + CharacterRenderer + her three worlds, composed exactly
// as her hub.js constructor does (hub.js lines 33-48). Nothing of hers is rewritten.
import {BirthDataStore} from '../../vendor/youniverse-games/birth/birthDataStore.js';
import {PlayerSeedStore} from '../../vendor/youniverse-games/seed/playerSeed.js';
import {GameStateStore} from '../../vendor/youniverse-games/state/gameState.js';
import {MorphEngine} from '../../vendor/youniverse-games/morph/morphEngine.js';
import {CharacterRenderer} from '../../vendor/youniverse-games/render/characterRenderer.js';
import {WorldAdapter} from '../../vendor/youniverse-games/world/worldAdapter.js';
import * as hdRpg from '../../vendor/youniverse-games/games/hdRpg.js';
import * as ichingStory from '../../vendor/youniverse-games/games/ichingStory.js';
import * as oraclesThread from '../../vendor/youniverse-games/games/oraclesThread/index.js';
import {chartToBirthRecord} from './birth-record.mjs';
export const HER_WORLDS=Object.freeze(['hd-rpg','iching-story','oracles-thread']);
export function createGamesHost({dataDir,chart,address}){const birthRecord=chartToBirthRecord(chart,{address});
 const birthStore=new BirthDataStore({[birthRecord.birthDataId]:birthRecord}),seedStore=new PlayerSeedStore(dataDir),gameStateStore=new GameStateStore(dataDir),
  morphEngine=new MorphEngine({seedStore}),renderer=new CharacterRenderer(dataDir),
  worlds=new WorldAdapter({seedStore,gameStateStore,morphEngine,renderer,birthStore,dataDir});
 worlds.registerWorld('iching-story',ichingStory.create);worlds.registerWorld('hd-rpg',hdRpg.create);worlds.registerWorld('oracles-thread',oraclesThread.create);
 return {birthRecord,birthStore,seedStore,gameStateStore,morphEngine,renderer,worlds,
  // the human enters with their own avatar: visualIdentity comes from HER ingestPhoto() on the avatar image they chose
  player(visualIdentity){const id='pl-'+birthRecord.birthDataId.slice(3,15)+'-'+visualIdentity.fingerprint.slice(0,8);
   return seedStore.exists(id)?seedStore.load(id):seedStore.create({birthRecord:birthStore.get(birthRecord.birthDataId),visualIdentity,playerId:id})}}}
