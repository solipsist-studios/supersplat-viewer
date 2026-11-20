import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';

const html = readFileSync(new URL('../src/index.html', import.meta.url), 'utf8');

// The standalone page is plain markup with inline scripts, so neither the type checker nor the
// linter sees them. A syntax error there (a duplicated `const`, say) leaves `window.sse` unset
// and the viewer never starts, with nothing but a console error to show for it.
describe('src/index.html', () => {
    it('has an inline bootstrap script that parses', () => {
        // the first module script has no imports, so it is valid as a function body
        const script = /<script type="module">([\s\S]*?)<\/script>/.exec(html)?.[1];
        assert.ok(script, 'no inline module script found');
        assert.doesNotThrow(() => new Function(script));
    });
});
