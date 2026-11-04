/* Browser integration: writes ONLY to synthetic ledgers; real pilot remains pending. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn,spawnSync} = require('node:child_process');
const {chromium} = require(path.resolve('../flir-pipeline-explorer/node_modules/@playwright/test'));
const python = path.resolve('.venv/Scripts/python.exe');
const output = path.resolve('reports/phase2b-browser-' + Date.now());
fs.mkdirSync(output,{recursive:true});
const synthetic = path.join(output,'synthetic');
const init = spawnSync(python,['-c',"import runpy,sys;runpy.run_path('tests/test_dependency_pilot.py')['synthetic_pilot'](sys.argv[1])",synthetic],{windowsHide:true,encoding:'utf8'});
assert.equal(init.status,0,init.stderr);
const children = new Set();
async function start(pilot,db,role='reviewer1'){
 const child=spawn(python,['-u','-c','import sys;from flir_pipeline.alignment.dependency_store import DependencyStore;from flir_pipeline.alignment.pilot_server import make_server;s=make_server(DependencyStore(sys.argv[1],sys.argv[2],sys.argv[3]),0,"dependency_ui.html");print("PORT="+str(s.server_port),flush=True);s.serve_forever()',pilot,db,role],{windowsHide:true,stdio:['ignore','pipe','pipe']});children.add(child);
 let logs='';child.stderr.on('data',d=>logs+=d.toString());
 const port=await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error(logs||'Service timeout')),30000);child.stdout.on('data',d=>{const m=d.toString().match(/PORT=(\d+)/);if(m){clearTimeout(timeout);resolve(Number(m[1]));}});child.on('exit',code=>{clearTimeout(timeout);reject(Error('Service exit '+code+': '+logs));});});
 return {child,base:'http://127.0.0.1:'+port};
}
async function stop(s){if(s&&s.child.exitCode===null){await new Promise(resolve=>{s.child.once('exit',resolve);s.child.kill();});children.delete(s.child);}}
async function loaded(page){await page.waitForFunction(()=>document.querySelectorAll('#content img').length>0&&[...document.querySelectorAll('#content img')].every(i=>i.complete&&i.naturalWidth>0));}
async function inspect(page,server,count){await page.goto(server.base);await page.waitForFunction(n=>document.querySelector('#cases').options.length===n,count);const keys=new Set();for(let i=0;i<count;i++){await page.selectOption('#cases',String(i));await loaded(page);for(const src of await page.locator('#content img').evaluateAll(xs=>xs.map(x=>new URL(x.src).pathname)))keys.add(src);}return keys.size;}
(async()=>{const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));const result={};let s;
try{
 const db=path.join(output,'synthetic-r1.sqlite3');s=await start(synthetic,db);result.syntheticImages=await inspect(page,s,4);await page.selectOption('#cases','0');await loaded(page);
 assert.equal(await page.locator('[data-group="evidence"]:checked').count(),0);
 await page.selectOption('#status','reviewed');await page.selectOption('#continuity','shot_change');await page.selectOption('#spatial','same_region');await page.selectOption('#confidence','medium');await page.fill('#reviewer','SYNTHETIC independent reviewer');await page.fill('#justification','Revisión sintética: región y toma son independientes.');
 await page.locator('button[type=submit]').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('evidence'));result.emptyEvidenceRejected=true;
 await page.locator('#bulk').click();assert.equal(await page.locator('[data-group="evidence"]:checked').count(),2);await page.locator('button[type=submit]').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('guardada'));
 const exported=await(await page.request.get(s.base+'/api/export/json')).text();await stop(s);s=await start(synthetic,db);await page.goto(s.base);await loaded(page);assert.equal(await(await page.request.get(s.base+'/api/export/json')).text(),exported);assert.equal(await page.locator('#justification').inputValue(),'Revisión sintética: región y toma son independientes.');result.persistenceAfterRestart=true;result.spanishRoundtrip=true;
 await stop(s);s=await start(synthetic,path.join(output,'synthetic-r2.sqlite3'),'reviewer2');await inspect(page,s,2);const blind=await(await page.request.get(s.base+'/api/pilot')).json();assert.deepEqual(blind.history,[]);assert.deepEqual(blind.adjudication_sources,[]);assert.equal((await page.request.get(s.base+'/media/m7')).status(),400);result.blindReviewer=true;await stop(s);
 const real=path.resolve(process.argv[2]||'reports/phase2b-final'),realDB=path.resolve('reports/phase2b-reviewer1.sqlite3');s=await start(real,realDB);const before=await(await page.request.get(s.base+'/api/export/json')).text();result.realImages=await inspect(page,s,24);await page.screenshot({path:path.join(output,'real-pilot.png'),fullPage:true});assert.equal(await(await page.request.get(s.base+'/api/export/json')).text(),before);result.realDecisions=JSON.parse(before).history.length;assert.equal(result.realDecisions,0);await stop(s);
 s=await start(real,realDB);assert.equal(await(await page.request.get(s.base+'/api/export/json')).text(),before);result.realPendingAfterRestart=true;await stop(s);
 s=await start(real,path.resolve('reports/phase2b-reviewer2.sqlite3'),'reviewer2');result.realSecondReviewerImages=await inspect(page,s,8);assert.equal((await(await page.request.get(s.base+'/api/pilot')).json()).history.length,0);assert.deepEqual(errors,[]);result.browserErrors=errors;
 fs.writeFileSync(path.join(output,'results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({output,...result}));
}finally{await stop(s);for(const child of children)child.kill();await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
