// Offline build helper. No KaTeX JavaScript is sent to the reader's browser.
const fs = require('node:fs');
const katex = require('./vendor/katex/katex.js');
const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = formulas.map(({tex, display}, index) => {
  try {
    return katex.renderToString(tex, {
      displayMode: display,
      output: 'htmlAndMathml',
      throwOnError: true,
      strict: 'error',
      trust: false,
      maxExpand: 1000,
    });
  } catch (error) {
    console.error(`formula ${index + 1} ${JSON.stringify(tex)}: ${error.message}`);
    process.exit(1);
  }
});
process.stdout.write(JSON.stringify(output));
