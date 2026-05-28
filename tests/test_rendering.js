'use strict';
/**
 * Rendering tests for renderAnswer() / escapeHtml().
 * Run with: node tests/test_rendering.js
 * Exit 0 = all pass, exit 1 = failures.
 */

// ---- Functions under test (copied from tenancy/static/app.js) ----

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderAnswer(text) {
  const idx = text.lastIndexOf('\n\nSources:');
  if (idx !== -1) text = text.substring(0, idx);
  text = escapeHtml(text.trim());

  const html = text.split(/\n{2,}/).map(para => {
    const lines = para.split('\n');

    if (lines.some(l => /^[-*] /.test(l.trim()))) {
      const items = [];
      let cur = null;
      for (const line of lines) {
        if (/^[-*] /.test(line.trim())) {
          if (cur !== null) items.push(cur);
          cur = line.trim().replace(/^[-*] /, '').replace(/  $/, '');
        } else if (cur !== null && line.trim()) {
          cur += ' ' + line.trim();
        }
      }
      if (cur !== null) items.push(cur);
      return `<ul>${items.map(t => `<li>${t}</li>`).join('')}</ul>`;
    }

    if (lines.some(l => /^\d+\. /.test(l.trim()))) {
      const items = [];
      let cur = null;
      for (const line of lines) {
        const m = line.trim().match(/^(\d+)\. (.*)/);
        if (m) {
          if (cur) items.push(cur);
          cur = { num: m[1], text: m[2].replace(/  $/, '') };
        } else if (cur && line.trim()) {
          cur.text += ' ' + line.trim();
        }
      }
      if (cur) items.push(cur);
      return `<ol>${items.map(it => `<li value="${it.num}">${it.text}</li>`).join('')}</ol>`;
    }

    return `<p>${lines.map(l => l.replace(/  $/, '')).join('<br>')}</p>`;
  }).join('');

  return html
    .replace(/\[S(\d+)\]/g, '<span class="citation">[S$1]</span>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// ---- Test harness ----

let passed = 0;
let failed = 0;

function assert(name, condition, actual) {
  if (condition) {
    console.log(`  PASS  ${name}`);
    passed++;
  } else {
    console.error(`  FAIL  ${name}`);
    if (actual !== undefined) console.error(`        got: ${JSON.stringify(actual)}`);
    failed++;
  }
}

function contains(html, fragment) {
  return html.includes(fragment);
}

function notContains(html, fragment) {
  return !html.includes(fragment);
}

// ---- Tests ----

console.log('\nescapeHtml');

assert('escapes ampersand',    escapeHtml('a & b') === 'a &amp; b');
assert('escapes less-than',    escapeHtml('<script>') === '&lt;script&gt;');
assert('escapes greater-than', escapeHtml('x > y') === 'x &gt; y');
assert('escapes double-quote', escapeHtml('"hi"') === '&quot;hi&quot;');
assert('leaves plain text',    escapeHtml('hello') === 'hello');

console.log('\nrenderAnswer - plain paragraphs');

{
  const h = renderAnswer('Hello world.');
  assert('wraps in <p>',         contains(h, '<p>Hello world.</p>'));
  assert('no stray <ul>/<ol>',   notContains(h, '<ul>') && notContains(h, '<ol>'));
}

{
  const h = renderAnswer('First para.\n\nSecond para.');
  assert('two paragraphs',       contains(h, '<p>First para.</p>') && contains(h, '<p>Second para.</p>'));
}

{
  const h = renderAnswer('Line one.\nLine two.');
  assert('single newline -> <br>', contains(h, 'Line one.<br>Line two.'));
}

console.log('\nrenderAnswer - trailing two-space line breaks stripped');

{
  const h = renderAnswer('Some text.  \nNext line.');
  assert('trailing spaces stripped from line', notContains(h, '  <br>'));
}

console.log('\nrenderAnswer - Sources block stripped');

{
  const h = renderAnswer('Answer text.\n\nSources:\n  [S1] Some case');
  assert('sources block removed',  notContains(h, 'Sources:'));
  assert('answer text kept',       contains(h, 'Answer text.'));
}

{
  const h = renderAnswer('Answer.\n\nSources:\n  [S1] Case A\n  [S2] Case B');
  assert('multi-source block removed', notContains(h, '[S1] Case A'));
  assert('answer before sources kept', contains(h, 'Answer.'));
}

console.log('\nrenderAnswer - bold and citations');

{
  const h = renderAnswer('This is **important**.');
  assert('bold rendered',     contains(h, '<strong>important</strong>'));
  assert('asterisks removed', notContains(h, '**'));
}

{
  const h = renderAnswer('See [S1] and [S2].');
  assert('[S1] citation span', contains(h, '<span class="citation">[S1]</span>'));
  assert('[S2] citation span', contains(h, '<span class="citation">[S2]</span>'));
}

{
  const h = renderAnswer('**Bold with [S3] citation.**');
  assert('bold + citation combined', contains(h, '<strong>') && contains(h, '<span class="citation">'));
}

console.log('\nrenderAnswer - bullet lists (dash)');

{
  const h = renderAnswer('- Item one\n- Item two\n- Item three');
  assert('ul produced',            contains(h, '<ul>'));
  assert('item one in li',         contains(h, '<li>Item one</li>'));
  assert('item two in li',         contains(h, '<li>Item two</li>'));
  assert('no ol for bullet list',  notContains(h, '<ol>'));
}

{
  const h = renderAnswer('* Asterisk bullet\n* Second');
  assert('asterisk bullet -> ul',  contains(h, '<ul>'));
  assert('asterisk item in li',    contains(h, '<li>Asterisk bullet</li>'));
}

console.log('\nrenderAnswer - bullet lists with continuation lines');

{
  // LLM sometimes wraps bullet body onto the next line
  const h = renderAnswer('- **Key point:**  \n  The body of this bullet.\n- Second item.');
  assert('continuation joined into li',  contains(h, 'The body of this bullet.'));
  assert('second item present',          contains(h, '<li>Second item.</li>'));
  assert('no orphaned continuation <p>', notContains(h, '<p>The body'));
}

console.log('\nrenderAnswer - numbered lists (single-line items)');

{
  const h = renderAnswer('1. First\n2. Second\n3. Third');
  assert('ol produced',           contains(h, '<ol>'));
  assert('value=1 on first item', contains(h, 'value="1"'));
  assert('value=2 on second',     contains(h, 'value="2"'));
  assert('value=3 on third',      contains(h, 'value="3"'));
  assert('no ul for numbered',    notContains(h, '<ul>'));
}

console.log('\nrenderAnswer - numbered lists preserve original numbers');

{
  // Each item in its own paragraph (LLM separates with blank lines)
  const h = renderAnswer('1. First item\n\n2. Second item\n\n4. Fourth item');
  assert('value=1 present', contains(h, 'value="1"'));
  assert('value=2 present', contains(h, 'value="2"'));
  assert('value=4 present', contains(h, 'value="4"'));
  assert('no value=3',      notContains(h, 'value="3"'));
}

console.log('\nrenderAnswer - numbered lists with multi-line items (real LLM pattern)');

{
  // The exact pattern that caused the original bug:
  // "1. **Question?**  \n   Answer text here [S1]."
  const input =
    '1. **Is the PM required to provide a remote?**  \n' +
    '   There is no clear legal obligation [S1].\n\n' +
    '2. **Can tenants go to body corporate?**  \n' +
    '   This depends on the property structure [S2].\n\n' +
    '3. **What is usual NZ practice?**  \n' +
    '   Responsibility varies case by case [S3].';
  const h = renderAnswer(input);
  assert('item 1: bold header present',      contains(h, '<strong>Is the PM required'));
  assert('item 1: body text present',        contains(h, 'There is no clear legal obligation'));
  assert('item 1: citation present',         contains(h, '<span class="citation">[S1]</span>'));
  assert('item 2: body text present',        contains(h, 'This depends on the property structure'));
  assert('item 3: body text present',        contains(h, 'Responsibility varies case by case'));
  assert('value="1" on item 1',             contains(h, 'value="1"'));
  assert('value="2" on item 2',             contains(h, 'value="2"'));
  assert('value="3" on item 3',             contains(h, 'value="3"'));
  assert('no orphaned <p> for body lines',  notContains(h, '<p>There is no clear'));
  assert('no orphaned <p> for item 2 body', notContains(h, '<p>This depends'));
}

console.log('\nrenderAnswer - numbered list items each in own paragraph block');

{
  // LLM sometimes puts each item in its own double-newline-separated block
  const input =
    '1. **Question one?**  \n   Answer to one.\n\n' +
    '2. **Question two?**  \n   Answer to two.';
  const h = renderAnswer(input);
  assert('item 1 body in output', contains(h, 'Answer to one.'));
  assert('item 2 body in output', contains(h, 'Answer to two.'));
}

console.log('\nrenderAnswer - XSS prevention');

{
  const h = renderAnswer('User said <script>alert(1)</script>');
  assert('script tag escaped',    notContains(h, '<script>'));
  assert('lt escaped',            contains(h, '&lt;script&gt;'));
}

{
  const h = renderAnswer('1. Item with <b>raw html</b>');
  assert('raw html in list escaped', notContains(h, '<b>'));
  assert('lt in list escaped',       contains(h, '&lt;b&gt;'));
}

{
  const h = renderAnswer('- Bullet with <img src=x onerror=alert(1)>');
  assert('img tag in bullet escaped', notContains(h, '<img'));
}

console.log('\nrenderAnswer - mixed content');

{
  const input = 'Intro paragraph.\n\n1. First point\n2. Second point\n\nConclusion paragraph.';
  const h = renderAnswer(input);
  assert('intro para present',   contains(h, '<p>Intro paragraph.</p>'));
  assert('ol present',           contains(h, '<ol>'));
  assert('conclusion present',   contains(h, '<p>Conclusion paragraph.</p>'));
}

{
  const input = 'Intro.\n\n- Bullet A\n- Bullet B\n\nOutro.';
  const h = renderAnswer(input);
  assert('intro before ul',  h.indexOf('<p>Intro') < h.indexOf('<ul>'));
  assert('ul before outro',  h.indexOf('</ul>') < h.indexOf('<p>Outro'));
}

// ---- Summary ----

console.log('\nrenderAnswer - streaming edge cases');

{
  // Empty string (e.g. stream ended before any tokens)
  const h = renderAnswer('');
  assert('empty string -> empty <p>',  h === '<p></p>' || h === '');
}

{
  // Leading/trailing newlines (LLM sometimes starts with blank lines)
  const h = renderAnswer('\n\nActual answer here.\n\n');
  assert('leading/trailing newlines trimmed', contains(h, '<p>Actual answer here.</p>'));
  assert('no empty <p> before content',     notContains(h, '<p></p>'));
}

{
  // Incomplete bold — unclosed ** (LLM cut off mid-token)
  const h = renderAnswer('This is **incomplete bold');
  assert('incomplete bold is safe HTML',   contains(h, '**incomplete bold'));
  assert('no dangling <strong> tag',       notContains(h, '<strong>incomplete bold'));
}

{
  // Unclosed citation bracket [S without closing ]
  const h = renderAnswer('See source [S1] and partial [S');
  assert('closed citation rendered',    contains(h, '<span class="citation">[S1]</span>'));
  assert('unclosed [S left as text',    contains(h, '[S'));
}

{
  // Only whitespace/newlines
  const h = renderAnswer('   \n\n   ');
  assert('whitespace-only -> empty output', h === '<p></p>' || h === '');
}

{
  // Single line with trailing \\n (common in streaming: last token is \\n)
  const h = renderAnswer('Answer text.\n');
  assert('trailing single newline handled', contains(h, 'Answer text.'));
}

{
  // Bold spanning multiple tokens (accumulated during streaming)
  const h = renderAnswer('The **key point** is important and **another** too.');
  assert('multiple bold spans rendered', 
    contains(h, '<strong>key point</strong>') && contains(h, '<strong>another</strong>'));
}

{
  // Numbered list with only one item (common in short answers)
  const h = renderAnswer('1. Only one item here.');
  assert('single numbered item -> ol',  contains(h, '<ol>'));
  assert('single item has value="1"',   contains(h, 'value="1"'));
}

console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
