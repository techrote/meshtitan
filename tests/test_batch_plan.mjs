import { test } from 'node:test';
import assert from 'node:assert/strict';
import { guidanceValues, planBatch } from '../src/meshtitan/web/batch.js';

const settings = { hunyuan_shape: {seed:100,guidance_scale:3},triposr:{mc_resolution:256} };
const assets = [{asset_id:'a',source_name:'a.png',output_format:'glb'},
  {asset_id:'b',source_name:'b.png',output_format:'glb'}];

test('multi-view combines labelled views before seed and guidance expansion',()=>{
  const mvAssets = assets.map((asset,i)=>({...asset,view:i?'back':'front'}));
  const mvSettings = {hunyuan_multiview:{seed:10,guidance_scale:4}};
  const args = {assets:mvAssets,backends:['hunyuan_multiview'],settings:mvSettings,variations:2,
    seedMode:'increment',sweep:{enabled:true,count:2,step:1},multiview:true};
  const jobs=planBatch(args);
  assert.equal(jobs.length,4);
  assert.deepEqual(jobs.map(job=>job.view_assets),Array(4).fill({front:'a',back:'b'}));
  assert.deepEqual(jobs.map(job=>job.parameters.seed),[10,10,11,11]);
  assert.ok(jobs.every(job=>job.asset_id==='a'));
  assert.throws(()=>planBatch({...args,assets:mvAssets.map(asset=>({...asset,view:'front'}))}));
  assert.throws(()=>planBatch({...args,assets:mvAssets.slice(0,1)}));
});
test('guidance sweep is a Cartesian product with fixed seeds within each comparison',()=>{
  const jobs=planBatch({assets,backends:['hunyuan_shape','triposr'],settings,variations:2,
    seedMode:'increment',sweep:{enabled:true,count:3,step:0.5}});
  assert.equal(jobs.length,14);
  assert.deepEqual(jobs.slice(0,6).map(job=>[job.parameters.seed,job.parameters.guidance_scale]),
    [[100,3],[100,3.5],[100,4],[101,3],[101,3.5],[101,4]]);
  assert.equal(jobs[6].backend,'triposr');
  assert.equal(jobs[6].asset_id,'a');
  assert.equal(jobs[7].parameters.seed,102);
  assert.equal(jobs[7].asset_id,'b');
  assert.equal(jobs.filter(job=>job.backend==='triposr').length,2);
});
test('random seeds are selected once per seed variation, not per guidance value',()=>{
  let seed=500;
  const jobs=planBatch({assets:assets.slice(0,1),backends:['hunyuan_shape'],settings,variations:2,
    seedMode:'random',sweep:{enabled:true,count:2,step:1},randomSeed:()=>seed++});
  assert.deepEqual(jobs.map(job=>job.parameters.seed),[500,500,501,501]);
});
test('sweeps reject invalid ranges, support descending values, and avoid floating point drift',()=>{
  assert.deepEqual(guidanceValues(3,{enabled:true,count:4,step:0.1}),[3,3.1,3.2,3.3]);
  assert.deepEqual(guidanceValues(3,{enabled:true,count:3,step:-1}),[3,2,1]);
  assert.deepEqual(guidanceValues(5,{enabled:false,count:99,step:99}),[5]);
  for(const sweep of [{enabled:true,count:17,step:1},{enabled:true,count:2,step:Infinity},
    {enabled:true,count:2,step:30},{enabled:true,count:4,step:-2}])
    assert.throws(()=>guidanceValues(3,sweep));
});
test('oversized Cartesian products are rejected and TripoSR is independent of Hunyuan sweeps',()=>{
  assert.throws(()=>planBatch({assets,backends:['hunyuan_shape'],settings,variations:8,
    seedMode:'fixed',sweep:{enabled:true,count:16,step:0.1}}),/200/);
  assert.equal(planBatch({assets,backends:['triposr'],settings,variations:8,
    seedMode:'fixed',sweep:{enabled:true,count:99,step:99}}).length,2);
});
