/* Synthetic writes only; real pilot is inspected through GET without annotations. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { chromium } = require(path.resolve('../flir-pipeline-explorer/node_modules/@playwright/test'));
const python = path.resolve('.venv/Scripts/python.exe');
const output = path.resolve('reports/phase2a1-browser-' + Date.now());
fs.mkdirSync(output, { recursive: true });
const synthetic = path.join(output, 'synthetic');
const syntheticDB = path.join(output, 'synthetic.sqlite3');
const real = path.resolve(process.argv[2] || 'reports/phase2a1-pilot');
const realDB = path.resolve(process.argv[3] || 'reports/phase2a1-reviews.sqlite3');
const init = spawnSync(python, ['-c', "import runpy,sys;runpy.run_path('tests/test_alignment_pilot.py')['synthetic_pilot'](sys.argv[1])", synthetic], { windowsHide: true, encoding: 'utf8' });
assert.equal(init.status, 0, init.stderr);
const children = new Set();
async function start(pilot, db) {
  const child = spawn(python, ['-u', '-c', 'import sys;from flir_pipeline.alignment.pilot_store import PilotStore;from flir_pipeline.alignment.pilot_server import make_server;s=make_server(PilotStore(sys.argv[1],sys.argv[2]),0);print("PORT="+str(s.server_port),flush=True);s.serve_forever()', pilot, db], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  children.add(child);
  let log = '';
  child.stderr.on('data', data => { log += data.toString(); });
  const port = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(Error('Service timeout: ' + log)), 30000);
    child.stdout.on('data', data => { const match = data.toString().match(/PORT=(\d+)/); if (match) { clearTimeout(timeout); resolve(Number(match[1])); } });
    child.on('exit', code => { clearTimeout(timeout); reject(Error('Service exited ' + code + ': ' + log)); });
  });
  return { child, base: 'http://127.0.0.1:' + port };
}
async function stop(server) {
  if (!server || server.child.exitCode !== null) return;
  await new Promise(resolve => { server.child.once('exit', resolve); server.child.kill(); });
  children.delete(server.child);
}
async function loaded(page) {
  await page.waitForFunction(() => document.querySelectorAll('#content img').length > 0 && [...document.querySelectorAll('#content img')].every(i => i.complete && i.naturalWidth > 0));
}
async function inspectAll(page, base) {
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#cases').options.length === 24);
  const keys = new Set();
  for (let i = 0; i < 24; i++) {
    await page.selectOption('#cases', String(i));
    await loaded(page);
    for (const src of await page.locator('#content img').evaluateAll(xs => xs.map(x => new URL(x.src).pathname))) keys.add(src);
  }
  return keys.size;
}
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  let server;
  const result = {};
  try {
    server = await start(synthetic, syntheticDB);
    result.syntheticImagesLoaded = await inspectAll(page, server.base);
    await page.selectOption('#cases', '0');
    await loaded(page);
    const snap = await (await page.request.get(server.base + '/api/pilot')).json();
    const c = snap.pilot.cases[0];
    await page.selectOption('#status', 'reviewed');
    await page.selectOption('#label', 'correct_match');
    await page.selectOption('#confidence', 'high');
    await page.fill('#reviewer', 'SYNTHETIC browser reviewer');
    await page.fill('#justification', 'Revisión sintética: árboles, señal y alternativas. No es una anotación FAC.');
    await page.getByRole('button', { name: 'Guardar revisión' }).click();
    await page.waitForFunction(() => document.querySelector('#message').textContent.includes('evidence'));
    assert.equal((await (await page.request.get(server.base + '/api/pilot')).json()).history.length, 0);
    for (const key of [c.original_media, c.alternatives[0].frame_media]) await page.locator('[data-media="' + key + '"]').check();
    await page.getByRole('button', { name: 'Guardar revisión' }).click();
    await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Versión 1'));
    await page.selectOption('#label', 'incorrect_match');
    await page.selectOption('#better', '2');
    await page.selectOption('#control', 'incorrect_match');
    for (const key of [c.alternatives[1].frame_media, c.control.media]) await page.locator('[data-media="' + key + '"]').check();
    await page.getByRole('button', { name: 'Guardar revisión' }).click();
    await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Versión 2'));
    const before = await (await page.request.get(server.base + '/api/export/json')).json();
    assert.equal(before.history.length, 2);
    assert.equal(before.history[0].human_verified_top1, true);
    assert.equal(before.history[1].human_verified_top1, false);
    await stop(server);
    server = await start(synthetic, syntheticDB);
    await page.goto(server.base);
    await page.waitForFunction(() => document.querySelector('#history').textContent.includes('decision_id'));
    assert.deepEqual(await (await page.request.get(server.base + '/api/export/json')).json(), before);
    assert.match(await page.inputValue('#justification'), /Revisión sintética/);
    assert.equal(await page.inputValue('#label'), 'incorrect_match');
    assert.equal(await page.inputValue('#better'), '2');
    const csv = await (await page.request.get(server.base + '/api/export/csv')).text();
    assert.match(csv, /initial_pending/);
    assert.match(csv, /incorrect_match/);
    await page.screenshot({ path: path.join(output, 'synthetic-review.png') });
    result.syntheticPersistence = { revisions: 2, restartExactExport: true, spanishRoundtrip: true, emptyEvidenceRejected: true };
    await stop(server);
    server = await start(real, realDB);
    const realBefore = await (await page.request.get(server.base + '/api/pilot')).json();
    result.realImagesLoaded = await inspectAll(page, server.base);
    assert.equal(result.realImagesLoaded, Object.keys(realBefore.pilot.media).length);
    await page.selectOption('#cases', '0');
    await loaded(page);
    await page.screenshot({ path: path.join(output, 'real-pilot.png') });
    await stop(server);
    server = await start(real, realDB);
    const realAfter = await (await page.request.get(server.base + '/api/pilot')).json();
    assert.deepEqual(realAfter.history, realBefore.history);
    result.realReviewHistoryCount = realAfter.history.length;
    result.realReadOnlyRestart = true;
    result.report = await (await page.request.get(server.base + '/api/report')).json();
    assert.equal(errors.length, 0, errors.join('\n'));
    result.runtimeErrors = errors;
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ output, ...result }, null, 2));
  } finally {
    await stop(server);
    await browser.close();
    for (const child of children) child.kill();
  }
})().catch(error => { console.error(error); for (const child of children) child.kill(); process.exit(1); });
