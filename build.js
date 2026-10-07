// node build.js spec.json out.html — embed a spec into viewer.html
const fs = require('fs'), path = require('path');
const [specPath, outPath] = process.argv.slice(2);
if (!specPath || !outPath) { console.error('usage: node build.js spec.json out.html'); process.exit(1); }
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
if (!spec.root || !spec.root.children) throw new Error('spec needs "root" with "children"');
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const html = fs.readFileSync(path.join(__dirname, 'viewer.html'), 'utf8')
  .replace(/(<script type="application\/json" id="spec">)[\s\S]*?(<\/script>)/, (_, a, b) => a + JSON.stringify(spec).replace(/</g, '\\u003c') + b)
  .replace(/<title>.*?<\/title>/, () => `<title>${esc(spec.title || 'Model Flow')}</title>`);
fs.writeFileSync(outPath, html);
console.log('wrote', outPath, (html.length / 1024).toFixed(0) + 'KB');
