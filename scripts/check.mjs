/**
 * Dependency-free checks for the logic that has no DOM in it: wake anchors,
 * the interruption gate, the persona identity lock and the agentic action
 * schema. Run with `npm test`.
 *
 * The worker-level behaviour (linting, sandbox capability limits, execution
 * timeouts, memory persistence) needs a real browser; see README "Verifying".
 */

import assert from 'node:assert/strict';
import { matchWakeAnchor, isInterruptToken, buildPersonaBlock, WAKE_ANCHORS, INTERRUPT_TOKENS } from '../src/utils/persona.js';
import { extractAction, validateAction, SchemaError, isFetchAllowed } from '../src/utils/schema.js';

let pass = 0;
const check = (name, fn) => {
  try { fn(); console.log(`  ok   ${name}`); pass += 1; }
  catch (e) { console.log(`  FAIL ${name}: ${e.message}`); process.exitCode = 1; }
};

console.log('\nwake anchors (SAD 2.1)');
check('"Hey Elias, what time is it" -> anchor + remainder', () => {
  const m = matchWakeAnchor('Hey Elias, what time is it');
  assert.equal(m.anchor.toLowerCase(), 'hey elias');
  assert.equal(m.remainder, 'what time is it');
});
check('bare "Elias" wakes', () => assert.ok(matchWakeAnchor('elias are you there')));
check('"Thorne!" wakes', () => assert.ok(matchWakeAnchor('Thorne! come in')));
check('"hey elias" beats the inner "elias"', () => {
  assert.equal(matchWakeAnchor('hey elias hello').anchor.toLowerCase(), 'hey elias');
});
check('substring does not false-trigger', () => assert.equal(matchWakeAnchor('eliasburg is nice'), null));
check('unrelated speech does not wake', () => assert.equal(matchWakeAnchor('turn off the lights'), null));

console.log('\ninterrupt gate (SAD 2.1)');
check('"stop" interrupts', () => assert.ok(isInterruptToken('stop')));
check('"wait" interrupts', () => assert.ok(isInterruptToken('wait a second')));
check('"STOP" is case-insensitive', () => assert.ok(isInterruptToken('STOP')));
check('"stopwatch" does not interrupt', () => assert.ok(!isInterruptToken('set a stopwatch')));
check('ordinary speech does not interrupt', () => assert.ok(!isInterruptToken('that sounds good')));

console.log('\npersona identity lock');
check('persona names Elias Thorne and the schema', () => {
  const p = buildPersonaBlock({ toolNames: ['weather'] });
  assert.ok(p.includes('Elias Thorne'));
  assert.ok(p.includes('create_tool'));
  assert.ok(p.includes('telemetryMessage'));
  assert.ok(p.includes('weather'));
});

console.log('\naction schema (SAD 2.3)');
check('extracts a fenced action with trailing prose', () => {
  const r = extractAction('ok\n```json\n{"action":"self_improve","toolName":"t","code":"return 1;","telemetryMessage":"Fixing t."}\n```\nthere');
  assert.equal(r.action.action, 'self_improve');
  assert.equal(r.prose, 'ok\n\nthere');
});
check('extracts a bare object with nested braces and strings', () => {
  const r = extractAction('{"action":"create_tool","toolName":"t","code":"return {a:\\"}\\"};","telemetryMessage":"Building t."}');
  assert.equal(r.action.toolName, 't');
});
check('returns null when there is no action', () => assert.equal(extractAction('just talking'), null));
check('rejects a bad toolName with the field named', () => {
  assert.throws(() => validateAction({ action: 'create_tool', toolName: '9bad name', code: 'x', telemetryMessage: 'hi there' }),
    (e) => e instanceof SchemaError && e.field === 'toolName');
});
check('rejects a missing telemetryMessage', () => {
  assert.throws(() => validateAction({ action: 'create_tool', toolName: 'ok', code: 'x' }),
    (e) => e.field === 'telemetryMessage');
});
check('rejects a non-allow-listed dependency', () => {
  assert.throws(() => validateAction({ action: 'create_tool', toolName: 'ok', code: 'x', telemetryMessage: 'building', dependencies: ['https://evil.example/m.js'] }),
    (e) => e.field === 'dependencies');
});
check('rejects http dependencies', () => {
  assert.throws(() => validateAction({ action: 'create_tool', toolName: 'ok', code: 'x', telemetryMessage: 'building', dependencies: ['http://esm.sh/m.js'] }),
    (e) => e.field === 'dependencies');
});
check('invoke_tool parses JSON args out of "code"', () => {
  const a = validateAction({ action: 'invoke_tool', toolName: 'sum', code: '{"numbers":[1,2]}', telemetryMessage: 'Running sum.' });
  assert.deepEqual(a.args, { numbers: [1, 2] });
});
check('invoke_tool wraps non-JSON args', () => {
  const a = validateAction({ action: 'invoke_tool', toolName: 'sum', code: 'london', telemetryMessage: 'Running sum.' });
  assert.deepEqual(a.args, { input: 'london' });
});
check('fetch allow-list accepts esm.sh over https only', () => {
  assert.ok(isFetchAllowed('https://esm.sh/x'));
  assert.ok(!isFetchAllowed('http://esm.sh/x'));
  assert.ok(!isFetchAllowed('https://evil.example/x'));
});

console.log(`\n${pass} checks passed`);
